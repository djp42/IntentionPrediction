using BayesNets
using Discretizers
using JLD

basepath = "./results/Test1/"
paths = [string(basepath,i) for i in 0:4]

logname = "test_log_BN.txt"
logfile = "$basepath$logname"

scorename = "test_BN_p_dists.txt"
#scorefile = "$basepath$i/$scorename"
moveCats = [1,2,3]
numNextMoves = length(moveCats)
useLaneType = true
if contains(basepath,".1")
  useLaneType = false
end
laneTypeIndex = 3
laneTypeEncodedLen = length([0,0,0,0])
numYawBins = 20

max_parents = 7

#truncate the log file
f = open(logfile, "w")
close(f)

function Distributions.fit(::Type{DiscreteBayesNet}, data::DataFrame, params::GreedyHillClimbing, indexOfOut::Int;
    ncategories::Vector{Int} = map!(i->infer_number_of_instantiations(data[i]), Array(Int, ncol(data)), 1:ncol(data)),
    )

    n = ncol(data)
    parent_list = map!(i->Int[], Array(Vector{Int}, n), 1:n)
    datamat = convert(Matrix{Int}, data)'
    score_components = bayesian_score_components(parent_list, ncategories, datamat, params.prior, params.cache)

    while true
        best_diff = 0.0
        best_parent_list = parent_list
        i = indexOfOut
        # 1) add an edge (j->i)
        if length(parent_list[i]) < params.max_n_parents
            for j in deleteat!(collect(1:n), parent_list[i])
                    if adding_edge_preserves_acyclicity(parent_list, j, i)
                        new_parents = sort!(push!(copy(parent_list[i]), j))
                        new_component_score = bayesian_score_component(i, new_parents, ncategories, datamat, params.prior, params.cache)
                    if new_component_score - score_components[i] > best_diff
                        best_diff = new_component_score - score_components[i]
                        best_parent_list = deepcopy(parent_list)
                        best_parent_list[i] = new_parents
                    end
                end
             end
        end

        # 2) remove an edge
        for (idx, j) in enumerate(parent_list[i])

            new_parents = deleteat!(copy(parent_list[i]), idx)
            new_component_score = bayesian_score_component(i, new_parents, ncategories, datamat, params.prior, params.cache)
            if new_component_score - score_components[i] > best_diff
                best_diff = new_component_score - score_components[i]
                best_parent_list = deepcopy(parent_list)
                best_parent_list[i] = new_parents
            end

            # 3) flip an edge
            new_parent_list = deepcopy(parent_list) # TODO: make this more efficient
            deleteat!(new_parent_list[i], idx)

            if adding_edge_preserves_acyclicity(new_parent_list, i, j)
                sort!(push!(new_parent_list[j], i))
                new_diff = bayesian_score_component(i, new_parent_list[i], ncategories, datamat, params.prior, params.cache) - score_components[i]
                new_diff += bayesian_score_component(j, new_parent_list[j], ncategories, datamat, params.prior, params.cache) - score_components[j]
                if new_diff > best_diff
                    best_diff = new_diff
                    best_parent_list = new_parent_list
                end
            end
        end

        if best_diff > 0.0
            parent_list = best_parent_list
            score_components = bayesian_score_components(parent_list, ncategories, datamat, params.prior, params.cache)
        else
            break
        end
    end
    # construct the BayesNet
    cpds = Array(DiscreteCPD, n)
    varnames = names(data)
    for j in 1:n
        name = varnames[j]
        parents = varnames[parent_list[j]]
        cpds[j] = fit(DiscreteCPD, data, name, parents, params.prior,
                      parental_ncategories=ncategories[parent_list[j]],
                      target_ncategories=ncategories[j])
    end
    BayesNet(cpds)
end

open(logfile, "a") do logf
  write(logf,"Done making discretizers\n")
  write(logf, string(now(),"\n"))
end
allFeatures = Dict()
allTargets = Dict()
allFids = Dict()
nFids = 0
println("Loading features/targets for BN")
open(logfile, "a") do logf
  write(logf,"Loading features/targets for BN\n")
  write(logf, string(now(),"\n"))
end
numFeatures = 0
for path in paths
    i = path[end]
    f = open("$path/featureSet")
    allFeatures[i] = readdlm(f)
    allFeatures[i] = allFeatures[i][1:500,:]
    close(f)

    f = open("$path/targetSet")
    allTargets[i] = readdlm(f)
    allTargets[i] = allTargets[i][1:500,:]
    close(f)

    if numFeatures == 0
        numFeatures = length(allFeatures[i][1,:])
    end
    f = open("$path/Fids")
    allFids[i] = readdlm(f)
    #allFids[i] = allFids[i][1:1000]
    close(f)
    nFids += length(allFids[i])
end
moveIndex = numFeatures+1

println("Done loading features/targets for BN")
println("NumFeatures == ", numFeatures)
open(logfile, "a") do logf
  write(logf,"Done loading features/targets for BN\n")
  write(logf, string(now(),"\n"))
end

typeToEdges = Dict(
    "Vy"=>[0.0,0.5,20,40,60],
    "Vx"=>[-1000,-20,-5,-0.5,0.5,5,20,1000],
    "A"=>[-5000,-20,-5,-0.5,0.5,5,20,5000],
    "dist" => [0.0, 5, 25, 100, 500, 10000],
    "hdwy" => [0.0, 5, 25, 100, 250, 10000],)
idToType = Dict(
    4=>"Vy",
    5=>"A",
    6=> "Vx",
    7=> "A",
    8=> "yaw",
    9=> "hdwy",
    10=> "dist",)
#above dict was made when lanetype was integer
indexToID = zeros(Int64, (1, numFeatures+1+laneTypeEncodedLen))   #+1 for when no lane type, shift needs buffer
coreLen = 9 + laneTypeEncodedLen
histLen = 9
numHist = 4
totHist = histLen * numHist
traffLen = 8
for index in 1:numFeatures
    ID = index
    if index <= coreLen
      ID = index
      if useLaneType
        if index >= 4
          ID = index - laneTypeEncodedLen + 1
        end
      end
    elseif index <= coreLen + totHist
      ID = ((index - coreLen) % histLen)
      if ID == 0
        ID = histLen
      end
      ID += 1
    else
      ID = ((index - (coreLen + totHist)) % traffLen)
      if ID == 0
        ID = traffLen
      elseif ID < 3
        ID = traffLen + 1
      end
      ID += 1
    end
    indexToID[index] = ID
end
println("Done making discretizer dictionaries")


#big for loop
for i in keys(allFeatures)
  #truncate score file
  f = open("$basepath$i/$scorename", "w")
  close(f)
  println("Starting feature/target formatting for CV #: $i")
  open(logfile, "a") do logf
    write(logf,"Starting feature/target formatting for CV #: $i \n")
    write(logf, string(now(),"\n"))
  end
  testlines = allFeatures[i]
  testactuals = allTargets[i]

  trainlines = zeros(Float64, (1,numFeatures))
  trainactuals = zeros(Float64, (1,numFeatures))
  first = true
  for j in keys(allFeatures)
    if j != i
      if first == true
        trainlines = allFeatures[j]
        trainactuals = allTargets[j]
        first = false
      else
        trainlines = vcat(trainlines, allFeatures[j])
        trainactuals = vcat(trainactuals, allTargets[j])
      end
    end
  end
  println("Done feature/target formatting for CV #: $i")
  println("Starting to discretize for CV #: $i")
  open(logfile, "a") do logf
    write(logf,"Done feature/target formatting for CV #: $i \n")
    write(logf,"Starting to discretize for CV #: $i \n")
    write(logf, string(now(),"\n"))
  end
  #make discretizers with all lines, outlier problems
  allLines = vcat(trainlines, testlines)
  allactuals = vcat(trainactuals, testactuals)
  discAllLines = zeros(Int64, (length(trainlines[:,1])+length(testlines[:,1]), numFeatures+1))
  discTrainlines = zeros(Int64, (length(trainlines[:,1]), numFeatures+1))
  discTestlines = zeros(Int64, (length(testlines[:,1]), numFeatures+1))
  #DiscreteAlready = [1,2,3,11,12,20,21,29,30]
  for k in 1:numFeatures
    id = indexToID[k]
    if haskey(idToType, id) #if not, already discretized
      F_type = idToType[id]
      if haskey(typeToEdges, F_type)
        Disc = LinearDiscretizer(typeToEdges[F_type])
      else
        Disc = LinearDiscretizer(binedges(DiscretizeUniformWidth(numYawBins),allLines[:,k]))
      end
      DiscretizedAll = encode(Disc, allLines[:,k])
      DiscretizedTrain = encode(Disc, trainlines[:,k])
      DiscretizedTest = encode(Disc, testlines[:,k])
    else
      DiscretizedAll = allLines[:,k]
      DiscretizedTrain = trainlines[:,k]
      DiscretizedTest = testlines[:,k]
    end
    discTrainlines[:,k] = encode(CategoricalDiscretizer(DiscretizedAll), DiscretizedTrain)
    discTestlines[:,k] = encode(CategoricalDiscretizer(DiscretizedAll), DiscretizedTest)
    discAllLines[:,k] = encode(CategoricalDiscretizer(DiscretizedAll), DiscretizedAll)
  end
  moveDiscretizer = CategoricalDiscretizer(trainactuals)
  discTrainlines[:,moveIndex] = encode(moveDiscretizer,trainactuals)
  discTestlines[:,moveIndex] = encode(moveDiscretizer,testactuals)
  discAllLines[:,moveIndex] = encode(moveDiscretizer,allactuals)
  println("Done discretizing for CV #: $i")
  open(logfile, "a") do logf
    write(logf,"Done discretizing for CV #: $i \n")
    write(logf, string(now(),"\n"))
  end
  alldata = convert(DataFrame, discAllLines) #used to find num bins
  traindata = convert(DataFrame, discTrainlines)
  testdata = convert(DataFrame, discTestlines)

  println("Starting to fit BN for CV #: $i")
  open(logfile, "a") do logf
    write(logf,"Starting to fit BN for CV #: $i \n")
    write(logf, string(now(),"\n"))
  end
  max_parents = min(numFeatures, max_parents)
  println("Max parents: ", max_parents)
  params = GreedyHillClimbing(ScoreComponentCache(traindata), max_n_parents=max_parents, prior=UniformPrior())
  num_bins_all = map!(i->infer_number_of_instantiations(alldata[i]), Array(Int, ncol(alldata)), 1:ncol(alldata))
  bnDis = @time fit(DiscreteBayesNet, traindata, params, moveIndex; ncategories=num_bins_all)
  println("Done fitting BN for CV #: $i")
  open(logfile, "a") do logf
    write(logf,"Done fitting BN for CV #: $i \n")
    write(logf, string(now(),"\n"))
    println("Starting to score BN for CV #: $i")
    write(logf,"Starting to score BN for CV #: $i\n")
    write(logf, string(now(),"\n"))
  end
  save(string("$basepath$i","/BN_model.jld"), "bnDis", bnDis)
  score = 0
  numNaN = 0
  nrows = length(testlines[:,1])
  moveCats = unique(collect(testdata[:,moveIndex]))
  numNextMoves = length(moveCats)
  for index in 1:nrows
      actualNextMove = convert(Int64,decode(moveDiscretizer,testdata[index,moveIndex]))
      p_dists = zeros(Float64, (1, numNextMoves))
      for move in moveCats
        featureline = testdata[index,:]
        featureline[moveIndex] = encode(moveDiscretizer, move)
        p_dists[move] = pdf(bnDis, featureline)
      end
      for x in p_dists
        if !(x > 1) && !(x < 1) && !(x == 1)
          p_dists = [1.0/numNextMoves for i in 1:numNextMoves]
          numNaN += 1
          break
        end
      end
      p_dists = p_dists ./ (sum(p_dists) + 0.0000000000000001)
      p_right = p_dists[actualNextMove]
      score += (1.0  - p_right)
      scoref = open("$basepath$i/$scorename", "a")
      writedlm(scoref, reshape(p_dists, (1,numNextMoves)))
      close(scoref)
  end
  println("Score:", score)
  println("Num Predicitons:", nrows)
  println("Num NaN:", numNaN)
  scoref = open("$basepath$i/$scorename", "a")
  #writedlm(scoref, score)
  #writedlm(scoref, numNaN)
  close(scoref)
  println("Done scoring BN for CV #: $i")
  open(logfile, "a") do logf
    write(logf,"Done scoring BN for CV #: $i \n")
    write(logf, string(now(),"\n"))
  end
end #end of for i in allfeatures keys
#end  #for log file
