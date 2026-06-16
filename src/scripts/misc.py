import matplotlib.pyplot as plt


def set_font(size: int = 25)-> None:
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "mathtext.fontset": "cm",
            "font.size": size
            })

def axis_ticks(ax):

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
        
        ax.tick_params(color='black', width=1.5)
        ax.tick_params(
            which='major',
            width=2,
            length=10,
            color='black',
            direction='in',
            top=True, bottom=True, left=True, right=True
        )
        ax.tick_params(
            which='minor',
            width=2,
            length=6,
            color='black',
            direction='in',
            top=True, bottom=True, left=True, right=True
    )

    
