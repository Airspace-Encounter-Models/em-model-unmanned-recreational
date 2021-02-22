# Old Networks

Original retrained networks created by S.M. Katz based on data and code from E.R. Mueller. These networks had a few small inconsistencies with E.R. Mueller's original 2016 paper. Specifically, rrHeaddot and rrHeaddot_tp1 were missing a bin in the CPDs, and one of the dependencies in the transition Bayes net was flipped from the diagram in the paper.

I originally did not have access to Eric's Bayes nets and had relearned all the parameters. I ended up getting access to the original networks, so I converted his networks to work with BayesNets.jl and replaced the these old ones with is original networks.