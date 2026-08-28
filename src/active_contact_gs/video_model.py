from __future__ import annotations

import torch
from torch import nn

from .amortized_model import AmortizedPhysicsPosterior, PosteriorOutput


def render_gaussian_video(
    poses: torch.Tensor,
    geometry: torch.Tensor,
    *,
    resolution: int = 16,
    selected_frames: int = 6,
) -> torch.Tensor:
    """Render pose trajectories as textured anisotropic Gaussian splats.

    The body is an oriented elliptical Gaussian. Four colored Gaussian texture
    anchors make orientation observable, as it would be from appearance features
    in an object-centric 3DGS. Coordinates are normalized by the known arena.
    """
    frame_count = poses.shape[-2]
    indices = torch.linspace(
        0, frame_count - 1, selected_frames, device=poses.device
    ).round().long()
    poses = poses.index_select(-2, indices)
    leading = poses.shape[:-2]
    poses = poses.reshape(-1, selected_frames, 4)
    expanded_geometry = geometry
    while expanded_geometry.ndim < len(leading) + 1:
        expanded_geometry = expanded_geometry.unsqueeze(-2)
    expanded_geometry = torch.broadcast_to(
        expanded_geometry, (*leading, 4)
    ).reshape(-1, 4)
    major = expanded_geometry[:, None, 0] / expanded_geometry[:, None, 3]
    minor = expanded_geometry[:, None, 1] / expanded_geometry[:, None, 3]
    center = poses[..., :2] / expanded_geometry[:, None, 3:4]
    sine, cosine = poses[..., 2], poses[..., 3]
    coordinates = torch.linspace(-1.0, 1.0, resolution, device=poses.device)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)
    difference = grid[None, None] - center[:, :, None, None]
    local_x = cosine[:, :, None, None] * difference[..., 0] + sine[:, :, None, None] * difference[..., 1]
    local_y = -sine[:, :, None, None] * difference[..., 0] + cosine[:, :, None, None] * difference[..., 1]
    body = torch.exp(
        -0.5
        * (
            (local_x / (0.72 * major[:, :, None, None]).clamp_min(0.015)).square()
            + (local_y / (0.72 * minor[:, :, None, None]).clamp_min(0.012)).square()
        )
    )
    # A weak neutral body texture leaves the four appearance anchors visible.
    # Their asymmetric RGB codes make both position and orientation observable
    # without exposing pose coordinates to the posterior.
    base_color = torch.tensor(
        [0.12, 0.12, 0.12], device=poses.device, dtype=poses.dtype
    )
    image = body[..., None] * base_color

    local_anchor = torch.stack(
        (
            torch.stack((0.58 * major, torch.zeros_like(major)), dim=-1),
            torch.stack((-0.58 * major, torch.zeros_like(major)), dim=-1),
            torch.stack((torch.zeros_like(minor), 0.58 * minor), dim=-1),
            torch.stack((torch.zeros_like(minor), -0.58 * minor), dim=-1),
        ),
        dim=-2,
    )
    anchor_x = cosine[..., None] * local_anchor[..., 0] - sine[..., None] * local_anchor[..., 1]
    anchor_y = sine[..., None] * local_anchor[..., 0] + cosine[..., None] * local_anchor[..., 1]
    anchors = center[..., None, :] + torch.stack((anchor_x, anchor_y), dim=-1)
    anchor_colors = torch.tensor(
        [
            [1.00, 0.05, 0.05],
            [0.05, 0.90, 0.90],
            [0.05, 1.00, 0.05],
            [0.05, 0.05, 1.00],
        ],
        device=poses.device,
        dtype=poses.dtype,
    )
    anchor_difference = grid[None, None, None] - anchors[..., None, None, :]
    marker = torch.exp(
        -0.5 * anchor_difference.square().sum(dim=-1) / (0.055**2)
    )
    image = image + torch.einsum("btqhw,qc->bthwc", marker, anchor_colors)
    image = image.clamp(0.0, 1.0).permute(0, 1, 4, 2, 3)
    return image.reshape(*leading, selected_frames, 3, resolution, resolution)


class GaussianVideoPosterior(AmortizedPhysicsPosterior):
    """Hybrid posterior whose probe encoder consumes only rendered images."""

    def __init__(
        self,
        *,
        observation_frames: int = 19,
        width: int = 192,
        heads: int = 6,
        layers: int = 4,
        dropout: float = 0.05,
        geometry_mode: str = "full",
        image_resolution: int = 32,
        video_frames: int = 6,
    ) -> None:
        super().__init__(
            observation_frames=observation_frames,
            width=width,
            heads=heads,
            layers=layers,
            dropout=dropout,
            geometry_mode=geometry_mode,
        )
        self.image_resolution = image_resolution
        self.video_frames = video_frames
        if image_resolution % 8:
            raise ValueError("image_resolution must be divisible by 8")
        self.frame_feature_dimension = 96 * (image_resolution // 8) ** 2
        self.moment_dimension = 3 * 6
        # Replace the pose-token encoder so checkpoints contain no unused path.
        self.probe_encoder = nn.Identity()
        self.frame_encoder = nn.Sequential(
            # CoordConv supplies absolute image location while all dynamical
            # evidence still originates in RGB pixels.
            nn.Conv2d(5, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
        )
        self.frame_projection = nn.Sequential(
            nn.Linear(self.frame_feature_dimension + self.moment_dimension, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.temporal_embedding = nn.Parameter(torch.empty(video_frames, width))
        nn.init.normal_(self.temporal_embedding, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, num_layers=2, enable_nested_tensor=False
        )
        self.temporal_norm = nn.LayerNorm(width)
        self.video_probe_encoder = nn.Sequential(
            nn.Linear(width + 4, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
        )

    def _pixel_features(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CoordConv input and RGB mass/centroid/covariance moments."""
        resolution = frames.shape[-1]
        coordinates = torch.linspace(
            -1.0, 1.0, resolution, device=frames.device, dtype=frames.dtype
        )
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        coordinate_channels = torch.stack((grid_x, grid_y))[None].expand(
            len(frames), -1, -1, -1
        )
        coordconv = torch.cat((frames, coordinate_channels), dim=1)

        weights = frames.float().clamp_min(0.0)
        mass = weights.sum(dim=(-2, -1)).clamp_min(1e-5)
        x = grid_x.float()[None, None]
        y = grid_y.float()[None, None]
        centroid_x = (weights * x).sum(dim=(-2, -1)) / mass
        centroid_y = (weights * y).sum(dim=(-2, -1)) / mass
        centered_x = x - centroid_x[..., None, None]
        centered_y = y - centroid_y[..., None, None]
        covariance_xx = (weights * centered_x.square()).sum(dim=(-2, -1)) / mass
        covariance_yy = (weights * centered_y.square()).sum(dim=(-2, -1)) / mass
        covariance_xy = (weights * centered_x * centered_y).sum(dim=(-2, -1)) / mass
        normalized_mass = torch.log1p(mass) / 8.0
        moments = torch.stack(
            (
                normalized_mass,
                centroid_x,
                centroid_y,
                covariance_xx,
                covariance_yy,
                covariance_xy,
            ),
            dim=-1,
        ).flatten(start_dim=1)
        return coordconv, moments.to(frames.dtype)

    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        geometry: torch.Tensor,
        probe_mask: torch.Tensor,
    ) -> PosteriorOutput:
        normalized_actions, _, encoded_geometry = self._normalize_inputs(
            actions, observations, geometry
        )
        with torch.no_grad():
            video = render_gaussian_video(
                observations,
                geometry[:, None],
                resolution=self.image_resolution,
                selected_frames=self.video_frames,
            )
        batch, probes = actions.shape[:2]
        frames = video.reshape(-1, 3, self.image_resolution, self.image_resolution)
        frame_features = []
        moment_features = []
        # Bounding the CNN chunk makes fantasy-EIG evaluation memory predictable.
        for start in range(0, len(frames), 4096):
            coordconv, moments = self._pixel_features(frames[start : start + 4096])
            frame_features.append(self.frame_encoder(coordconv))
            moment_features.append(moments)
        frame_features_tensor = torch.cat(frame_features)
        moment_features_tensor = torch.cat(moment_features)
        frame_tokens = self.frame_projection(
            torch.cat((frame_features_tensor, moment_features_tensor), dim=-1)
        ).reshape(batch * probes, self.video_frames, -1)
        frame_tokens = frame_tokens + self.temporal_embedding[None]
        temporal_tokens = self.temporal_encoder(frame_tokens)
        video_summary = self.temporal_norm(temporal_tokens.mean(dim=1)).reshape(
            batch, probes, -1
        )
        probe_tokens = self.video_probe_encoder(
            torch.cat((normalized_actions, video_summary), dim=-1)
        )
        summary = self.summary_token.expand(batch, -1, -1)
        geometry_token = self.geometry_encoder(encoded_geometry).unsqueeze(1)
        tokens = torch.cat((summary, geometry_token, probe_tokens), dim=1)
        prefix_mask = torch.zeros((batch, 2), device=actions.device, dtype=torch.bool)
        padding_mask = torch.cat((prefix_mask, ~probe_mask.to(torch.bool)), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        summary = self.final_norm(encoded[:, 0])
        hypothesis_logits = self.hypothesis_head(summary)
        parameter_raw = self.parameter_head(summary)
        parameter_mean = torch.sigmoid(parameter_raw[:, :8].reshape(-1, 2, 4))
        parameter_std = 0.010 + 0.36 * torch.sigmoid(
            parameter_raw[:, 8:].reshape(-1, 2, 4)
        )
        return PosteriorOutput(hypothesis_logits, parameter_mean, parameter_std)
