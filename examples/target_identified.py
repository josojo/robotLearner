# ruff: noqa: F821
sim.observe(cameras=["overview", "wrist"])
sim.run_skill("identify_tie")
sim.observe(cameras=["overview", "wrist"])
