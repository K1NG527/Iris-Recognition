"""
Deep Iris Biometric Matcher.
Provides cosine similarity metric matching and multi-shift roll compensation
for verification (1:1) and identification (1:N).
"""

import numpy as np
import torch
import torch.nn.functional as F


class DeepIrisMatcher:
    """
    Biometric comparator for IrisDeepNet embeddings.
    
    Uses spherical cosine similarity space where genuine pairs yield high positive
    similarity (e.g. 0.60 to 0.95) and impostor pairs cluster near zero (~0.0 to 0.15).
    """
    def __init__(self, model=None, device="cpu", roll_shifts=(-8, -4, 0, 4, 8)):
        self.model = model
        self.device = device
        self.roll_shifts = roll_shifts
        if self.model is not None:
            self.model.eval()
            self.model.to(device)

    @torch.no_grad()
    def extract_embedding(self, tensor_img, multi_shift=False):
        """
        Extracts unit-normalized embedding for an input tensor (1, 1, 64, 512).
        If `multi_shift=True`, evaluates multiple circular angular shifts and returns
        either a stacked tensor or the primary embedding.
        """
        if self.model is None:
            raise RuntimeError("Matcher initialized without an IrisDeepNet model instance.")

        tensor_img = tensor_img.to(self.device)
        if not multi_shift or len(self.roll_shifts) <= 1:
            emb = self.model.extract_features(tensor_img)
            return emb.cpu()

        # Multi-shift roll test for rotation invariance
        shifted_tensors = []
        for shift in self.roll_shifts:
            shifted = torch.roll(tensor_img, shifts=shift, dims=-1)
            shifted_tensors.append(shifted)

        batch_shifted = torch.cat(shifted_tensors, dim=0) # (K, 1, 64, 512)
        embs = self.model.extract_features(batch_shifted) # (K, D)
        return embs.cpu()

    @staticmethod
    def compute_similarity(emb_a, emb_b):
        """
        Computes cosine similarity between two embeddings or embedding sets.
        Returns value in [-1.0, 1.0].
        """
        if isinstance(emb_a, np.ndarray):
            emb_a = torch.from_numpy(emb_a)
        if isinstance(emb_b, np.ndarray):
            emb_b = torch.from_numpy(emb_b)

        if emb_a.dim() == 1:
            emb_a = emb_a.unsqueeze(0)
        if emb_b.dim() == 1:
            emb_b = emb_b.unsqueeze(0)

        emb_a = F.normalize(emb_a, p=2, dim=-1)
        emb_b = F.normalize(emb_b, p=2, dim=-1)

        sim_matrix = torch.mm(emb_a, emb_b.t())
        return float(torch.max(sim_matrix).item())

    @staticmethod
    def compute_similarity_matrix(query_embs, gallery_embs):
        """
        Computes full cosine similarity matrix between N queries and M gallery templates.
        
        Parameters:
        - query_embs: (N, D) torch tensor or numpy array
        - gallery_embs: (M, D) torch tensor or numpy array
        
        Returns:
        - sim_matrix: (N, M) matrix of cosine similarities
        """
        if isinstance(query_embs, np.ndarray):
            query_embs = torch.from_numpy(query_embs)
        if isinstance(gallery_embs, np.ndarray):
            gallery_embs = torch.from_numpy(gallery_embs)

        q_norm = F.normalize(query_embs, p=2, dim=1)
        g_norm = F.normalize(gallery_embs, p=2, dim=1)

        sim_matrix = torch.mm(q_norm, g_norm.t())
        return sim_matrix.cpu().numpy()
