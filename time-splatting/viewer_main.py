import sys, time
from pathlib import Path
from dataclasses import dataclass, field
import viser
from gsplat_viewer import GsplatRenderTabState, GsplatViewer
import train
from gsplat.strategy import DefaultStrategy, MCMCStrategy
import options

if __name__=="__main__":
    port = 8080

    cfg = options.TimeSplattingConfig(strategy         = DefaultStrategy(reset_every=100000, verbose=True),
                                      shading_strategy = DefaultStrategy(reset_every=100000, verbose=True))
    cfg.data_dir = "/dataset"
    runner = train.Runner(cfg)

    print("AAAAA")

    viewer = GsplatViewer(
        server     = viser.ViserServer(port=port, verbose=False),
        render_fn  = runner._viewer_render_fn,
        output_dir = "/dataset",
        mode       = "training"
    )
    time.sleep(3000)