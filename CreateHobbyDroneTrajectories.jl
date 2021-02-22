# S. M. Katz modified code from E. R. Mueller
# Most code is kept in its original form
# Data was reprocessed using BayesNets.jl since I did not have the original final bayes net files
# Final Bayes Nets are stored in iBN_data.jld2 and tBN_data.jld2
# See the following citation and Eric's PhD thesis for more information about this model
# E. R. Mueller and M. J. Kochenderfer, 
# “Simulation comparison of collision avoidance algorithms for small multi-rotor aircraft,” 
# in AIAA Modeling and Simulation Technologies Conference, 2016, p. 3674

################################################################################################
############## THIS MIGHT HAVE A UNITS ISSUE SO CHECK HERE WHEN THINGS GO WRONG ################
mutable struct IntruderState
	altAGL::Float64
	altdot::Float64
	heading::Float64
	rrHeading::Float64
	rrHeaddot::Float64
	range::Float64
	vel::Float64   		# Note that this variable corresponds to dxdt in the BNs and the feature name
	vdot::Float64  		# Note that this variable corresponds to d2xdt2 in the BNs and the feature name
	x::Float64
	y::Float64
	x0::Float64
	y0::Float64
	xdd::Float64
	ydd::Float64
	t::Float64

	IntruderState(altAGL, altdot, heading, rrHeading, rrHeaddot, range, vel, vdot, x, y, x0, y0, xdd, ydd, t) = 
			  new(altAGL, altdot, heading, rrHeading, rrHeaddot, range, vel, vdot, x, y, x0, y0, xdd, ydd, t)
end

function IntruderState(;
	altAGL::Float64=0., 
	altdot::Float64=0., 
	heading::Float64=0.,
	rrHeading::Float64=0.,
	rrHeaddot::Float64=0.,
	range::Float64=0.,
	vel::Float64=0.,
	vdot::Float64=0.,
	x::Float64=0.,
	y::Float64=0.,
	x0::Float64=0.,
	y0::Float64=0.,
	xdd::Float64=0.,
	ydd::Float64=0.,
	t::Float64=0.)

	IntruderState(altAGL, altdot, heading, rrHeading, rrHeaddot, range, vel, vdot, x, y, x0, y0, xdd, ydd, t)
end

mutable struct BNParams
	# Holds the discretized bin sizes for the Bayes net.  It would be nice to be able to read this in directly from the xdsl file,
	# but it doesn't appear to be stored there.

	binDisc::Dict

	BNParams(binDisc) = new(binDisc)

end

function BNParams()
# This discretization serves two functions: allows us to bin the continuous data before using Kyle's program (if we use that), and
# do continuous sampling within a bin once a set of discrete bins has been selected from the BN.  To do the latter, we need outer limits
# on the bin sizes, which adds two values to each discretization.  During the former binning, the first and last elements of each of
# these arrays is ignored and -Inf and +Inf are used instead (so no data is left out). 

# A limitation of GeNIe is that discretization has to happen from zero, it cannot "wrap around".  This is a problem for states that
# represent angles because, for example, we'd like to bin everything from -22.5 to 22.5 in one bin, 22.5 to 67.5 in the second, etc.
# I should be able to add this capability myself if I do the binning.

	binDisc = Dict(
		:range     => [0., 11., 29., 101., 822., 10000.],
		:rrHeading => [0., 45., 90., 135., 180., 225., 270., 315., 360],
		:rrHeaddot =>  [-50., -8.698, -0.0324, 0.032, 8.079, 50.],
		:altAGL    => [0., 10., 50., 100., 400., 5000.],
		:altdot    => [-5, -0.3921, -0.060, 0.061, 0.41447, 5],
		:dxdt      =>  [0., 0.03, 0.10, 0.30, 1.75, 40.],
		:d2xdt2    => [-5., -0.46, -5.11e-6, 4.906e-5, 0.464, 5.],  # This may need to be narrower for the 2nd and 5th bins.
		:d2xdt2_tp1    => [-5., -0.46, -5.11e-6, 4.906e-5, 0.464, 5.],
		:altdot_tp1    => [-5, -0.3921, -0.060, 0.061, 0.41447, 5],
		:rrHeaddot_tp1 =>  [-50., -8.698, -0.0324, 0.032, 8.079, 50.])

		return BNParams(binDisc)
end
################################################################################################
################################################################################################

# Sample initial state distribution to obtain an initial state
function sampleBayesInitial(iBN::BayesNet; x0::Float64=0., y0::Float64=0., evidenceVal::Assignment=Assignment(), bnp::BNParams=BNParams(),
							maxAttempts::Int64=1000, bearingInit::Float64=0., VERBOSE::Bool=false)
	# The initial heading is not part of the bayes net, and shouldn't affect the dynamics or interactions with other aircraft, so a 
	# default value of 0 will be used.  
	# Function returns a continuous sample of the bayes net, optionally accepting set values for specific parameters in evidenceVal.
	# All variables coming out of this model will be in ft, ft/s, or ft/s^2.

	# x0, y0, x, y, xd, yd, xdd, ydd, h, hd = sampleBayesInitial(iBN)
	# x0, y0, x, y, xd, yd, xdd, ydd, h, hd = sampleBayesInitial(iBN, x0=0., y0=0., evidenceVal=Assignment(), bnp=BNParams(), maxAttempts=1000, bearingInit=0.)


	# If there is evidence, set that in the bayes net.  First have to get bins:
	evidenceBins = convertVals2Bins(iBN, evidenceVal, bnp=bnp)

	# Sample the network, getting bins.
	initSamp = rand(iBN, LikelihoodWeightedSampler(evidenceBins), 1) # rand_table_weighted(iBN, numSamples=1, consistentWith=evidenceBins)
	ind = 1
	while (size(initSamp,1)==0) && (ind<maxAttempts)
		initSamp = rand_table(iBN, numSamples=1, consistentWith=evidenceBins)
		ind+=1
	end
	if size(initSamp,1)==0
		if VERBOSE
			display("Could not find initial sample consistent with evidence.  Returning random sample.")
		end
		ind = 1
		while (size(initSamp,1)==0) && (ind<maxAttempts)
			initSamp = rand_table(iBN, numSamples=1)
			ind+=1
		end
	end

	# Convert the bin assignments to a table of boundaries for the bins
	binSampleBounds = convertBins2Bounds(iBN, initSamp, bnp=bnp)

	# Pull a continuous sample from the bins.
	contSample = disc2contSample(binSampleBounds)

	# Need to do the unit conversion right after sampling from the BN so that our state conversion equations are more
	# straightforward.  This should be cleaner becaues only the BN will be in aviation units, everything else will
	# be done in ft, ft/s, (check on altitude rate, make sure it's not in ft/min), radians.
	contSampleBasic = convertAvUnits2Basic(contSample)

	# Convert the sample from the existing variables to the starter variables
	x, y, xd, yd, xdd, ydd, h, hd = convertRRVars2EuclVars(contSampleBasic, x0=x0, y0=y0, bearingInit=bearingInit)

	# To test with a specific initial condition:
	# x=0.
	# y=0.
	# xd=0.
	# yd=0.
	# xdd=0.
	# ydd=0.
	# h=10.
	# hd=0.

	return x0, y0, x, y, xd, yd, xdd, ydd, h, hd

end

# Convert values to discretized bins
function convertVals2Bins(bn::BayesNet, evidenceVal::Assignment; bnp::BNParams=BNParams())

	evidenceBin = Assignment()

	for key in keys(evidenceVal)
		binInd=1
		while (binInd<length(bnp.binDisc[key])) && (evidenceVal[key]>bnp.binDisc[key][binInd+1])
			binInd+=1
		end
		# If the value is larger than the last boundary in the discretization, the value will be binned in the last bin.
		binInd = clamp(binInd,1,length(bnp.binDisc[key])-1) #clamp(binInd,1, length(domain(bn,key).elements))
		# evidenceBin[key] = bn.domains[bn.index[key]].elements[binInd]    # This line was for when indexing into an integer
		#****** I think the way I implemented the discretization, this will work ********
		evidenceBin[key] = binInd #bn.nodes[bn.name_to_index[key]].domain.elements[binInd]

		# evidenceBin[key] = binInd
	end

	return evidenceBin

end


function convertBins2Bounds(bn::BayesNet, samp::DataFrame; bnp::BNParams=BNParams())
	# Given a dataframe with a sample, look up the bin boundaries that correspond to the bin names (strings) that
	# comprise the domains of the BayesNet.  The nodes of the BayesNet, bn, will be labeled with the same symbol
	# as the dataframe columns.  The row of samp with data will have strings that are in the domains of the 
	# appropriate node in the BN. 

	j,n = size(samp)
	if j == 0
		display("Zero samples in bin->bound conversion, exiting.")
		return -1
	end

	if j>1
		display("More than one sample in bin->bound conversion, will only return the bins of the first: $(samp)")
		samp = samp[1,:]
	end

	binBounds = DataFrame()
	for key in names(samp)
		key = Symbol(key)
		if key!=:p  	# This is a probability key generated by the likelihood-weighted sampling process.  Shouldn't need it.
			bins = bnp.binDisc[key]   # This is an array of floats with the bin boundaries for key (one of the bin labels)
			binVal = samp[!, key][1]  	  # This is the text string label for the bin in the sample corresponding to the above bin discretization
			#indBin = find(bn.domains[bn.index[key]].elements.==binVal)[1]   # This is the bin index corresponding to the string label
			
			# SMK: I just need to find the bin index that is the bottom of the bin (maybe I already have this from the way I discretized?) 
			# - I think I do ******
			binInd = binVal #find(bn.nodes[bn.name_to_index[key]].domain.elements.==binVal)[1]

			#binVal = samp[key][1]
			#binInd = find(domain(bn,key).elements.==binVal)[1]
			binBounds[!, key] = bins[binInd:binInd+1]
		end
	end

	return binBounds

end

function disc2contSample(binBounds::DataFrame)

	contSamp = DataFrame()
	for key in names(binBounds)
		l,h = binBounds[!, key]
		contSamp[!, key] = [(h-l)*rand() + l]
	end

	return contSamp

end

################### Will be interesting to see if there are issues here ################
function convertAvUnits2Basic(samp::DataFrame)

	sampConv = DataFrame()
	for key in names(samp)

		if key == :rrHeaddot
			sampConv[!, key] = pi/180 * samp[!, key]
		elseif key == :rrHeading
			sampConv[!, key] = pi/180 * samp[!, key]
		elseif key == :rrHeaddot_tp1
			sampConv[!, key] = pi/180 * samp[!, key]
		else
			sampConv[!, key] = samp[!, key]
		end
	end

	return sampConv
end

# Converts bayes net sampled params to actual variables needed to propagate the trajectory
function convertRRVars2EuclVars(contSample::DataFrame; x0::Float64=0., y0::Float64=0., bearingInit::Float64=0.)
	# bearingInit in degrees

	R = contSample[!, :range][1]
	rrHeading = contSample[!, :rrHeading][1]
	rrHeaddot = contSample[!, :rrHeaddot][1]
	v = contSample[!, :dxdt][1]
	vd = contSample[!, :d2xdt2][1]

	x = x0 + R*sin(bearingInit)
	y = y0 + R*cos(bearingInit)

	xd = v*sin(bearingInit+rrHeading)
	yd = v*cos(bearingInit+rrHeading)

	bearingdot = (xd*(y-y0)-yd*(x-x0))/R^2

	xdd = vd*sin(bearingInit+rrHeading) + v*(rrHeaddot+bearingdot)*cos(bearingInit+rrHeading)
	ydd = vd*cos(bearingInit+rrHeading) - v*(rrHeaddot+bearingdot)*sin(bearingInit+rrHeading)

	h = contSample[!, :altAGL][1]
	hd = contSample[!, :altdot][1]

	return x, y, xd, yd, xdd, ydd, h, hd 

end

# Sets intruder state
function setInitState(x0::Float64, y0::Float64, x::Float64, y::Float64, xd::Float64, yd::Float64, xdd::Float64, ydd::Float64, 
					  h::Float64, hd::Float64; t::Float64=0.)
# Will probably just want to specify a few states and then calculate all the others.  This function will do that.
	
	dx = x-x0
	dy = y-y0
	R = sqrt(dx^2+dy^2)

	psi_t = atan(dx, dy)
	psi_t_dot = 0.
	if R!=0
		psi_t_dot = (xd*dy - yd*dx)/R^2
	end

	psi_h = atan(xd, yd)
	psi_h_dot = 0.
	if (xd^2+yd^2)!=0.
		psi_h_dot = (xdd*yd - ydd*xd)/(xd^2+yd^2)
	end

	psi_v = mod(psi_h - psi_t, 2*pi)
	psi_v_dot = psi_h_dot - psi_t_dot

	v = sqrt(xd^2+yd^2)
	v_dot = sqrt(xdd^2 + ydd^2)

	IntruderState(h, hd, psi_h, psi_v, psi_v_dot, R, v, v_dot, x, y, x0, y0, xdd, ydd, t)	
end

function trajComplete(iState::IntruderState; tMax::Float64=120)
	# For now, will just simulate to a max time

	if iState.t>= tMax
		return true
	end

	return false
end

function sampleBayesTransition(tBN::BayesNet, iState::IntruderState; bnp::BNParams=BNParams(), maxAttempts::Int64=1000, VERBOSE::Bool=false)

	# Set the given evidence (pulls out the relevant state from iState, sets bins)
	#display("$iState")
	evidenceBins = setTransitionEvidence(tBN, iState, bnp=bnp)

	# Pull a sample from network, getting bins
	tSamp = rand(tBN, LikelihoodWeightedSampler(evidenceBins), 1) #rand_table_weighted(tBN, numSamples=1, consistentWith=evidenceBins)
	ind=1
	while (size(tSamp,1)==0) && (ind<maxAttempts)
		tSamp = #rand_table_weighted(tBN, numSamples=1, consistentWith=evidenceBins)
		ind+=1
	end
	if size(tSamp,1)==0
		if VERBOSE
			display("Could not find transition sample consistent with evidence.  Returning random sample.")
		end
		ind = 1
		while (size(tSamp,1)==0) && (ind<maxAttempts)
			tSamp = rand_table_weighted(tBN, numSamples=1)
			ind+=1
		end
	end

	# Go through each bin, comparing whether it is a different bin from the last bin.  If it's the same, resample a continuous point
	# with probability 1-epsilon (epsilon ~ 0.02-0.08, or learn that value from the data). If its a different bin, just sample agoin.
	# As a first implementation I could just resample every time.

	# Convert the bin assignments to a table of boundaries for the bins
	binSampleBounds = convertBins2Bounds(tBN, tSamp, bnp=bnp)

	# Pull a continuous sample from the bins.
	contSample = disc2contSample(binSampleBounds)

	# Check whether the bin contains zero. If it does, return exactly zero.  Note that this will operate on all fields, while
	# we only care about zeroing out the t+1 fields (altdot, rrHeaddot and vdot).  Implementation will be cleaner if we 
	# zero out every field and then just return the appropriate ones.
	zeroOutSpanningBins!(contSample, binSampleBounds)

	# Need to do the unit conversion right after sampling from the BN so that our state conversion equations are more
	# straightforward.  This should be cleaner becaues only the BN will be in aviation units, everything else will
	# be done in ft, ft/s, (check on altitude rate, make sure it's not in ft/min), radians.
	contSampleBasic = convertAvUnits2Basic(contSample)

	hdot_t2 = contSampleBasic[!, :altdot_tp1][1]
	vdot_t2 = contSampleBasic[!, :d2xdt2_tp1][1]
	psivdot_t2 = contSampleBasic[!, :rrHeaddot_tp1][1]

	return hdot_t2, vdot_t2, psivdot_t2

end

function setTransitionEvidence(tBN::BayesNet, iState::IntruderState; bnp::BNParams=BNParams())
# this function sets the seven relevant feature values in iState into a dictionary, returning a mapping from
# features to feature values (not bins).  These feature

	evidenceVal = Assignment()

	evidenceVal[:altAGL] = iState.altAGL
	evidenceVal[:altdot] = iState.altdot
	evidenceVal[:d2xdt2] = iState.vdot
	evidenceVal[:rrHeaddot] = iState.rrHeaddot
	evidenceVal[:range] = iState.range
	evidenceVal[:dxdt] = iState.vel
	evidenceVal[:rrHeading] = iState.rrHeading

	convertVals2Bins(tBN, evidenceVal, bnp=bnp)

end

function zeroOutSpanningBins!(sample, binSampleBounds)

	for key in names(sample)
		if (maximum(binSampleBounds[!, key])>0.) & (minimum(binSampleBounds[!, key])<0.)
			sample[!, key] = [0.]
		end
	end

end

function getNextState(intState::IntruderState, hdot_t2::Float64, vdot_t2::Float64, psivdot_t2::Float64, dt::Float64; 
					  psiTol::Float64=1e-5, psidotTol::Float64=1e-6, maxIters::Int64=20, VERBOSE::Bool=false)
# Given a current state and the values of the d/dt terms (h, v, psi) at the next time step, return
# a state that represents the next state.  Note that psivdot is the range-relative heading rate, a pretty wierd
# state, but one that hopefully captures the necessary behavior from the simulated trajectories.
	
	# Basic state updates:
	h_t1 = intState.altAGL
	hdot_t1 = intState.altdot
	v_t1 = intState.vel
	vdot_t1 = intState.vdot
	psiv_t1 = intState.rrHeading
	psivdot_t1 = intState.rrHeaddot

	h_t2 = h_t1 + 0.5*(hdot_t1 + hdot_t2)*dt
	v_t2 = v_t1 + 0.5*(vdot_t1 + vdot_t2)*dt
	psiv_t2 = mod(psiv_t1 + 0.5*(psivdot_t1 + psivdot_t2)*dt, 2*pi)

	# If v_t2 < 0 then strange things are going to happen.  My assumption is that velocity is always positive, but the
	# Bayes net could certainly sample a "deceleration" (negative vdot_t2) when velocity is low, which would indeed
	# cause a negative velocity.  I could either just set the velocity to zero in that case, or take the absolute value
	# of the velocity and rotate the heading and rrheading vectors 180 degrees.  Would have ot figure out what to do
	# with psivdot_t2
	if v_t2 < 0. 
		v_t2=0.
		vdot_t2=0. 
	end
	t1 = intState.t

	# Derived variable state updates:
	x_t1 = intState.x
	y_t1 = intState.y
	x0 = intState.x0
	y0 = intState.y0
	psih_t1 = intState.heading
	xd_t1 = v_t1*sin(psih_t1)
	yd_t1 = v_t1*cos(psih_t1)
	xdd_t1 = intState.xdd
	ydd_t1 = intState.ydd
	v_t1==0. ? psihdot_t1=0. : psihdot_t1 = (xdd_t1*yd_t1 - ydd_t1*xd_t1)/(v_t1^2)

	# Initialize variables for iterative psihdot search (plus other variables)
	psih_t2 = psih_t1
	psihdot_t2 = psihdot_t1
	psih_old = Inf
	psihdot_old = Inf
	iters = 0
	xdd_t2 = xdd_t1
	ydd_t2 = ydd_t1
	xd_t2 = xd_t1
	yd_t2 = yd_t1
	x_t2 = x_t1
	y_t2 = y_t1
	R_t2 = sqrt(x_t1^2+y_t1^2)
	while ((abs(psih_t2-psih_old) > psiTol) | (abs(psihdot_t2-psihdot_old) > psidotTol)) & (iters<maxIters)
		xdd_t2 = vdot_t2*sin(psih_t2) + v_t2*psihdot_t2*cos(psih_t2)
		ydd_t2 = vdot_t2*cos(psih_t2) - v_t2*psihdot_t2*sin(psih_t2)
		xd_t2 = xd_t1 + 0.5*(xdd_t1+xdd_t2)*dt
		yd_t2 = yd_t1 + 0.5*(ydd_t1+ydd_t2)*dt
		x_t2 = x_t1 + 0.5*(xd_t1+xd_t2)*dt
		y_t2 = y_t1 + 0.5*(yd_t1+yd_t2)*dt
		R_t2 = sqrt(x_t2^2+y_t2^2)

		psih_old = psih_t2
		psihdot_old = psihdot_t2
		psih_t2 = atan(xd_t2,yd_t2)
		#	psihdot_t2 = (xdd_t2*yd_t2 - ydd_t2*xd_t2)/(xd_t2^2+yd_t2^2)  # Try the following instead?:
		R_t2==0. ? psitdot_t2 = 0. : psitdot_t2 = (xd_t2*(y_t2-y0) - yd_t2*(x_t2-x0))/R_t2^2
		psihdot_t2 = psivdot_t2-psitdot_t2

		# if VERBOSE
		# 	display("After iteration $(iters+1), error on heading is $(psih_t2-psih_old) and error on heading rate is $(psihdot_t2-psihdot_old)")
		# end

		iters+=1
	end
	t2 = t1+dt
	

	IntruderState(h_t2, hdot_t2, psih_t2, psiv_t2, psivdot_t2, R_t2, v_t2, vdot_t2, x_t2, y_t2, intState.x0, intState.y0, xdd_t2, ydd_t2, t2)

end