from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import wroclaw_taxonomy

class Plotter:
    """
    A class used to plot dendrite calculated by Wroclaw taxonomy method.

    ----------

    Attributes
    ----------
    obj : Dendrite
        dendrite object after using .calculate() method
    plot_style : dict
        dictionary of Matplotlib styling options for maps 

    Methods
    ----------
    set_style(style:dict = 'default')
        sets style of plots and animations created by this class
    plot(level:int = None, lines:bool = True, style:dict = None, show:bool = True)
        plots result dendrite and objects
    animate(out_file:str = 'dendrite.gif', frame_duration:int = 1, lines:bool = True, style:dict = None)
        created an animation presenting each step of dendrite creation
    """

    def __init__(self, dendrite:wroclaw_taxonomy.Dendrite, style:dict = None):
        """
        Parameters
        ----------
        dendrite : Dendrite
            dendrite object after using .calculate() method
        style : dict:
            dictionary of Matplotlib styling options for maps 
        """
        if dendrite._processed is False:
            raise AttributeError('Dendrite was not processed! Use .calculate() method first')
        else:
            self.obj = dendrite

        self.plot_style = {
            "markersize": 10,
            "cmap": 'jet',
            "line_color": '#222222',
            "object_color": '#ff0000'
        }
        if style is not None:
            self.set_style(style)
    
    def __str__(self):
        return f'<Plotter object:   default style>'
    
    def set_style(self, style: dict | str = 'default') -> Plotter:
        """
        Sets style for every map that will be generated later.

        ----------

        Parameters
        ----------
        style : dict
            dictionary containing style configuration of maps, e.g. markersize, cmap, line color and object color
        """
        if style != 'default':
            self.plot_style = style #| self.plot_style
        else:
            self.plot_style = {
                "markersize": 10,
                "cmap": 'jet',
                "line_color": '#222222',
                "object_color": '#ff0000'
            }
        return self
    
    def plot(self, level:int = None, lines:bool = True, style:dict = None, show:bool = True):
        """
        Displays map of computed dendrite and source objects.

        ----------

        Parameters
        ----------
        level : int
            only connections from smaller or equal level will be displayed on a map
        lines : bool
            if True, dendrite is plotted; if False, only source objects are plotted
        style : dict
            style configuration of map, e.g. markersize, cmap, line color and object color
        show : bool
            if True, map is displayed immediately; if False, map is returned and can be saved to variable
        """
        if style is None:
            style = self.plot_style
        else:
            style = self.plot_style | style
        dendrite = self.obj.dendrite
        objects = self.obj.results
        if level is not None:
            dendrite = dendrite[dendrite['level'] <= level]
        fig, ax = plt.subplots(figsize = (10, 10))
        if lines==True:
            for lvl, lwd in zip(range(1, max(dendrite['level']) + 1), np.arange(0.5, 2 + (1.5 / (max(dendrite['level']) + 1)), (1.5 / (max(dendrite['level']) + 1)))):
                dendrite[dendrite['level'] == lvl].plot(ax=ax, color=style["line_color"],  linewidth=lwd, zorder=5)

        if objects.geom_type[0] == 'Point' and level is not None:
            objects.plot(
                ax=ax,
                cmap=style["cmap"],
                markersize=style["markersize"],
                zorder=10,
                column=f'cluster{level}'
            )
        elif objects.geom_type[0] == 'Point' and level is None:
            objects.plot(
                ax=ax,
                color=style["object_color"],
                zorder=10,
                markersize=(objects['connections'] - 0.75) * 2
            )
        elif objects.geom_type[0] == 'MultiPolygon' and level is not None:
            objects.plot(
                ax=ax,
                cmap=style["cmap"],
                zorder=1,
                column=f'cluster{level}'
            )
        elif objects.geom_type[0] == 'MultiPolygon' and level is None:
            objects.plot(
                ax=ax,
                zorder=1,
                cmap='Reds',
                column='connections'
            )
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if show==True:
            plt.show()
        else:
            return fig

    def animate(self, out_file:str = 'dendrite.gif', frame_duration:int = 1, lines:bool = True, style:dict = None) -> str:
        """
        Creates GIF animation showing each step of creating dendrite (map of every level of dendrite connections).

        ----------

        Parameters
        ----------
        out_file : str
            path to output .gif file where animation will be saved
        frame_duration : int
            time of display of each frame in seconds
        lines : bool
            if True, dendrite is included; if False, only source objects are plotted
        style : dict
            style configuration of animation, e.g. markersize, cmap, line color and object color
        """
        dendrite = self.obj.dendrite
        n_frames = np.max(dendrite["level"].unique())
        files = []
        frames = []
        for i in range(1, n_frames + 1):
            img = self.plot(level=i, lines=lines, style=style, show=False)
            plt.close()
            img.savefig(f'frame{i}.png')
            files.append(f'frame{i}.png')
            frames.append(imageio.imread(f'frame{i}.png'))
        
        imageio.mimsave(out_file, frames, duration=frame_duration)
        for file in files:
            os.remove(file)

        return f"GIF saved in {out_file}"
