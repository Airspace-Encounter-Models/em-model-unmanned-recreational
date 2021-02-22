# Written by S. M. Katz to interface with E. R. Mueller hobby drone model

using BayesNets
using DataFrames
using CSV
using JLD2
using LightGraphs
using Random

include("CreateHobbyDroneTrajectories.jl")

# Load the BayesNets (note these are loaded in pieces because JLD2 was being dumb)
@load "iBN_data_v2.jld2"
dag = SimpleDiGraph{Int64}(ne, fadjlist, badjlist)
iBN = BayesNet(dag, cpds, name_to_index)

@load "tBN_data_v2.jld2"
dag = SimpleDiGraph{Int64}(ne, fadjlist, badjlist)
tBN = BayesNet(dag, cpds, name_to_index)

#########################################################################
# NOTE: EVERYTHING FROM HOBBY DRONE MODEL IS IN FT, FT/S, AND FT/S²
#########################################################################

"""
Main function to get an intruder trajectory
"""
function generate_HD(initBN::BayesNet, tranBN::BayesNet; 
					  psiTol::Float64=1e-6, psidotTol::Float64=1e-8, maxIters::Int64=20, VERBOSE::Bool=false, dt::Float64=0.2, 
					  tMax::Float64=120., bnp::BNParams=BNParams(), x0::Float64=0., y0::Float64=0., bearingInit::Float64=0., 
					  initEvidenceVal::Assignment=Assignment(), maxAttempts::Array{Int64}=[10000,100], t0::Float64=0.)
# stHistory = simulateIntruderTraj(iBN, tBN);

	stHistory = IntruderState[]
	t = 0.

	# Init state: sample from the initial bayes net:
	iState = setInitState(sampleBayesInitial(initBN, x0=x0, y0=y0, evidenceVal=initEvidenceVal, bnp=bnp,
											 maxAttempts=maxAttempts[1], bearingInit=bearingInit, VERBOSE=VERBOSE)..., t=t0)
	stHistory = [iState]

	while ~trajComplete(iState, tMax=tMax)

		# Sample from bayes net to get the next state variables
		hdot_t2, vdot_t2, psivdot_t2 = sampleBayesTransition(tranBN, iState, bnp=bnp, maxAttempts=maxAttempts[2], VERBOSE=VERBOSE)

		# integrate next state variables to get the complete state
		iState = getNextState(iState, hdot_t2, vdot_t2, psivdot_t2, dt, psiTol=psiTol, psidotTol=psidotTol, maxIters=maxIters, VERBOSE=VERBOSE)

		# record the state
		stHistory = [stHistory; iState]

	end

	return stHistory
end

# Get matrix of positions for state history
function convert_to_position(stHistory::Array{IntruderState})
	p = zeros(length(stHistory), 3)
	for i = 1:length(stHistory)
		p[i,:] = [stHistory[i].x, stHistory[i].y, stHistory[i].altAGL] # Should take care of units
	end
	return p
end

function generate_trajectory_file(dt::Float64, filename::String; tMax=120.0)
	stHistory = generate_HD(iBN, tBN, dt=dt, tMax=tMax)
	p = convert_to_position(stHistory)
	times = collect(range(0, step = dt, length = length(p[:,1])))
	df = DataFrame(time_s = times, x_ft = p[:,1], y_ft = p[:,2], z_ft = p[:,3])
	CSV.write(filename, df)
end

function generate_trajectory_file(dt::Float64, filename::String, num_trajs::Int64; tMax=120.0)
	xs = []
	ys = []
	zs = []
	times = []
	for i = 1:num_trajs
		stHistory = generate_HD(iBN, tBN, dt=dt, tMax=tMax)
		p = convert_to_position(stHistory)
		times = [times; collect(range(0, step = dt, length = length(p[:,1])))]
		xs = [xs; p[:,1]]
		ys = [ys; p[:,2]]
		zs = [zs; p[:,3]]
	end
	df = DataFrame(time_s = times, x_ft = xs, y_ft = ys, z_ft = zs)
	CSV.write(filename, df)
end