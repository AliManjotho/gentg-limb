import torch
from models.common.embeddings import JointEmbedding, LimbEmbedding

def test_joint_embedding_output_shape():
    B,T,J = 2, 7, 5
    k2d = torch.randn(B,T,J,2)
    emb = JointEmbedding(in_dim=2, d_model=32)
    out = emb(k2d)
    assert out.shape == (B,T,J,32)

def test_limb_embedding_output_shape():
    B,T,J = 2, 7, 5
    xyz = torch.randn(B,T,J,3)
    limbs = [(0,1),(1,2),(2,3)]
    emb = LimbEmbedding(d_model=32)
    out = emb(xyz, limbs)
    assert out.shape == (B,T,len(limbs),32)
