
# https://gallery.pyecharts.org/#/README


import random

import pyecharts
from pyecharts import options
from pyecharts.charts import Bar, Line, Grid
from pyecharts.charts import Page, Timeline


def DrawBar(x, height):
    '''
    '''
    _bar = (
        Bar()
        .add_xaxis(x)
        .add_yaxis('heightA', height)
        .add_yaxis('heightB', height)
        .set_global_opts(title_opts=options.TitleOpts(title='test'), 
                        xaxis_opts=options.AxisOpts(name='label', 
                                                    name_rotate=45, 
                                                    axislabel_opts={'rotate': 30})
                        )
    )
    return _bar


if __name__ == '__main__':

    page = Page()
    timeline = Timeline()
  

    for a in list('AB'):
        bar = DrawBar(x=list('abcde'), height=[random.randint(4, 10) for i in range(5)])
        timeline.add(bar, a)
    
    page.add(bar)
    page.add(timeline)

    page.render()

    
