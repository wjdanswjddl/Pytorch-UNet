import matplotlib.pyplot as plt

def log_fig(true_img, pred_img, vval=1):
    fig, ax = plt.subplots(figsize=(8, 5))
    true_fig = plt.imshow(true_img.T, 
                          aspect="auto", origin="lower", vmin=-vval, vmax=vval, cmap="seismic")
    plt.xlabel("wire")
    plt.ylabel("tick")
    plt.title("True")
    plt.colorbar()

    fig, ax = plt.subplots(figsize=(8, 5))
    pred_fig = plt.imshow(pred_img.T, 
                          aspect="auto", origin="lower", vmin=-vval, vmax=vval, cmap="seismic")
    plt.xlabel("wire")
    plt.ylabel("tick")
    plt.title("Prediction")
    plt.colorbar()

    return true_fig.get_figure(), pred_fig.get_figure()


def ep_fig(test_perf_p0, test_perf_p1, test_tags):

    fig_eff, ax = plt.subplots(figsize=(6,4))
    plt.plot(test_perf_p0.loc[0,:], 'o-', label="Plane 0")
    plt.plot(test_perf_p1.loc[0,:], 'o-', label="Plane 1")
    plt.xlabel("$\\theta_{xz}$ [degrees]", size=12)
    plt.ylabel("Pixel Efficiency", size=12)
    plt.ylim(0.4, 1)
    plt.xticks(range(len(test_tags)), test_tags)
    plt.legend(ncol=2, fontsize=10)
    plt.grid(linestyle="--")

    fig_pur, ax = plt.subplots(figsize=(6,4))
    plt.plot(test_perf_p0.loc[1,:], 'o-', label="Plane 0")
    plt.plot(test_perf_p1.loc[1,:], 'o-', label="Plane 1")
    plt.xlabel("$\\theta_{xz}$ [degrees]", size=12)
    plt.ylabel("Pixel Purity", size=12)
    plt.ylim(0.8, 1.02)
    plt.xticks(range(len(test_tags)), test_tags)
    plt.legend(ncol=2, fontsize=10)
    plt.grid(linestyle="--")
    
    return fig_eff.get_figure(), fig_pur.get_figure()

