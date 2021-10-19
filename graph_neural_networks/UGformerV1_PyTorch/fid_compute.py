import networkx as nx
from scipy import linalg
import numpy as np
import torch
## Coming from https://github.com/mseitzer/pytorch-fid
def compute_FID(mu1, mu2, cov1, cov2, eps = 1e-6):
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert cov1.shape == cov2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(cov1) +
            np.trace(cov2) - 2 * tr_covmean)

def compute_fid(ref_graph,ref_label,pred_graph,model):
    device = 0
    num_layers = 5
    num_mlp_layers = 2
    hidden_dim = 64
    final_dropout = 0.5
    graph_pooling_type = "sum"
    neighbor_pooling_type = "sum"
    learn_eps = False
    
    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    #model = GraphCNN(num_layers, num_mlp_layers, train_graphs[0].node_features.shape[1], hidden_dim, 5, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, 'cpu').to('cpu')
    #model.load_checkpoint()
    #model.eval()


    with torch.no_grad():
        embed_graphs_ref = model.get_activation(ref_graph)
        embed_graphs_ref=embed_graphs_ref.cpu().detach().numpy()
        mu_ref = np.mean(embed_graphs_ref, axis = 0)
        cov_ref = np.cov(embed_graphs_ref, rowvar = False)

        embed_graphs_pred = model.get_activation(pred_graph)
        embed_graphs_pred=embed_graphs_pred.cpu().detach().numpy()
        mu_pred = np.mean(embed_graphs_pred, axis = 0)
        cov_pred = np.cov(embed_graphs_pred, rowvar = False)

    fid = compute_FID(mu_ref,mu_pred,cov_ref,cov_pred)
    return fid

def test_acc(model, device, train_graphs):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc = float(correct / float(len(train_graphs)))


    print("accuracy : %f" % (acc))

    return acc