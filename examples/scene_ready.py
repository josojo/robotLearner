# ruff: noqa: F821
sim.observe(cameras=["work", "closeup", "overview", "wrist"])
sim.run_skill("settle")
sim.observe(cameras=["overview"])
