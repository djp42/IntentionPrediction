using BayesNets
using Discretizers
using JLD

basepath = "./results/ByIntersection/"
doingPeach = false 
testnums = ["111","011"]#"000","001","010","011", "100"]
intersections = collect(1:9)
test_intersections = collect(1:9)
doSubtest = false #2500 features only

function makePaths(basepath, testnums, intersections)
    paths = []
    for testnum in testnums
        for intersection in intersections
            push!(paths, string(basepath, testnum, "/", intersection))
        end
    end
    return paths
end

paths = makePaths(basepath, testnums, intersections)

typeToEdges = Dict(
    "V"=>[0.0,0.5,20,40,60],
    "A"=>[-5000,-20,-5,-0.5,0.5,5,20,5000],
    "dist" => [0.0, 5, 25, 100, 500, 10000],
    "hdwy" => [0.0, 5, 25, 100, 250, 10000],)
idToType = Dict(
    3=>"V",
    4=>"A",
    5=> "yaw",
    #6 is indicator
    7=> "hdwy",
    8=> "dist",
)


#returns allFeatures, allTargets dictionaries wheren "testnum/intersection" are the keys
function loadAllFeaturesTargets(paths)
    allFeatures = Dict()
    allTargets = Dict()
    allFids = Dict()
    nFids = 0
    numFeatures = 0
    for path in paths #path is result/ByIntersection/yyy/x
        i = path[end-4:end]
        f = open("$path/featuresAndTargets")
        allFeatures[i] = readdlm(f)
        allTargets[i] = allFeatures[i][:,end]  #nextMove is the last thing only
        allFeatures[i] = allFeatures[i][:,1:end-3]
        
        if doSubtest
            allFeatures[i] = allFeatures[i][1:2500,:]
            allTargets[i] = allTargets[i][1:2500,:]
        end
        close(f)
        
        if numFeatures == 0
            numFeatures = length(allFeatures[i][1,:])
        end
    end
    return allFeatures, allTargets, numFeatures
end

function makeDiscretizedThings(trainlines, testlines, trainactuals, testactuals, numFeatures,
                                           indexToID, numYawBins, moveIndex)
  allLines = vcat(trainlines, testlines)
  allactuals = vcat(trainactuals, testactuals)
  discAllLines = zeros(Int64, (length(trainlines[:,1])+length(testlines[:,1]), numFeatures+1))
  discTrainlines = zeros(Int64, (length(trainlines[:,1]), numFeatures+1))
  discTestlines = zeros(Int64, (length(testlines[:,1]), numFeatures+1))
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
      #println("index ", k, " has no id, aka already disc")
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

  alldata = convert(DataFrame, discAllLines) #used to find num bins
  traindata = convert(DataFrame, discTrainlines)
  testdata = convert(DataFrame, discTestlines)
  return alldata, traindata, testdata, moveDiscretizer
end

function makeIndexToID(numFeatures, testnum, laneTypeEncodedLen)
    useLaneType = (testnum[1] == '1')
    useHist = (testnum[2] == '1')
    useTraffic = (testnum[3] == '1')
    println(testnum, useLaneType, useHist, useTraffic)
    indexToID = zeros(Int64, (1, numFeatures+1+laneTypeEncodedLen))   #+1 for when no lane type, shift needs buffer
    coreLen = 8
    if useLaneType
        coreLen += laneTypeEncodedLen
    end
    histLen = 9 #indicator is first
    numHist = 4
    totHist = histLen * numHist
    traffLen = 8
    #base = lane, lane, v, a, yaw, 1{hdwy}, hdwy, dist
    #hist is the same, 4 times
    #traffic is dx, dy, v, a, yaw, 1{hdwy}, hdwy
    for index in 1:numFeatures
        ID = index
        if index <= coreLen
          if useLaneType
            if index > 4
              ID -= laneTypeEncodedLen
            else #laneTypeEncoding features are 0 ID
              ID = 0 
            end
          end
          ID += 1 #because all others get shifted, limits amount of confusion i think
        elseif useHist == false
          ID = ((index - coreLen) % traffLen)
          if ID == 0
            ID = traffLen
          elseif ID == 2 || ID == 3
            ID = 9 #distance + 1
          end
        elseif useHist && index <= coreLen + totHist
            ID = ((index - coreLen) % histLen)
            if ID == 0
                ID = histLen
            end
        else  #use hist and in traffic section
          ID = ((index - (coreLen + totHist)) % traffLen)
          if ID == 0
            ID = traffLen
          elseif ID == 2 || ID == 3
            ID = 9 #distance + 1
          end
        end
        ID -= 1 #shift for indicator at front of each traff
        indexToID[index] = ID
    end
    #for i in 1:numFeatures
    #    id = indexToID[i]
    #    println("index: ",i, "id: ", id)
    #    if haskey(idToType, id) 
    #        println("key:", idToType[id])
    #    end
    #end
    return indexToID
end

# fit function --- modified from https://github.com/sisl/BayesNets.jl/blob/master/src/DiscreteBayesNet/greedy_hill_climbing.jl
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
      parent_list[i] = collect(1:n)
      break
      #for i in 1:n #this is added for full fitting
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
                new_diff = bayesian_score_component(i, new_parent_list[i], ncategories, datamat, params.prior, 
                                                       params.cache) - score_components[i]
                new_diff += bayesian_score_component(j, new_parent_list[j], ncategories, datamat, params.prior, 
                                                       params.cache) - score_components[j]
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
      #end   #this is for the "for i in 1:n"
    end
    # construct the BayesNet
    cpds = Array(DiscreteCPD, n)
    varnames = names(data)
    for j in 1:n
        name = varnames[j]
        parents = varnames[parent_list[j]]
        cpds[j] = Distributions.fit(DiscreteCPD, data, name, parents, params.prior,
                      parental_ncategories=ncategories[parent_list[j]],
                      target_ncategories=ncategories[j])
    end
    #println(cpds)
    println("Parent List: ", parent_list)
    BayesNet(cpds)
end


#big for loop
println("Loading features/targets for BN")
allFeatures, allTargets, numFeatures = loadAllFeaturesTargets(paths)
println("Done loading features/targets for BN")
println("NumFeatures == ", numFeatures)
for inter in test_intersections
  for testnum in testnums
      i = string(testnum, "/", inter)
      scorename = "Ypred_BN"
      scorePath = string("$basepath",testnum, "/TestOn$inter","/$scorename")
      f = open(scorePath, "w")
      #truncate score file
      close(f)
      moveCats = [1,2,3]
      numNextMoves = length(moveCats)
      useLaneType = false
      if testnum[1] == "1"
        useLaneType = true
      end
      laneTypeIndex = 3
      laneTypeEncodedLen = length([0,0,0,0])
      numYawBins = 7
      #truncate the log file
      numFeatures = length(allFeatures[i][1,:])  
      indexToID = makeIndexToID(numFeatures, testnum, laneTypeEncodedLen)
      
      println("index to ID dict: \n ",indexToID)
      
      moveIndex = numFeatures+1

      println("Done making discretizer dictionaries")
      println("Starting feature/target formatting for CV #: $i")
      testlines = allFeatures[i]
      testactuals = allTargets[i]

      trainlines = zeros(Float64, (1,numFeatures))
      trainactuals = zeros(Float64, (1,numFeatures))
      first = true
      for other_inter in intersections
        if other_inter != inter  #the train lines are all that are not test
          j = string(testnum, "/", other_inter)
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
      alldata, traindata, testdata, moveDiscretizer = makeDiscretizedThings(trainlines, testlines, trainactuals, testactuals, 
                                                           numFeatures, indexToID, numYawBins, moveIndex)

      println("Done discretizing for CV #: $i")
      println("Starting to fit BN for CV #: $i")
      max_parents = numFeatures
      #max_parents = min(numFeatures, max_parents)
      println("Max parents: ", max_parents)

      params = GreedyHillClimbing(ScoreComponentCache(traindata), max_n_parents=max_parents, prior=UniformPrior())
      println("Done fitting params, starting to fit BN")
      
      num_bins_all = map!(i->infer_number_of_instantiations(alldata[i]), Array(Int, ncol(alldata)), 1:ncol(alldata))
      bnDis = @time fit(DiscreteBayesNet, traindata, params, moveIndex; ncategories=num_bins_all)
      println("Done fitting BN for CV #: $i")
      save(string("$basepath$i","/BN_model.jld"), "bnDis", bnDis)
      score = 0
      numNaN = 0
      nrows = length(testlines[:,1])
      moveCats = unique(collect(testdata[:,moveIndex]))
      numNextMoves = 3#length(moveCats)
      numRight = 0
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
          p_dists = p_dists ./ sum(p_dists)
          p_right = p_dists[actualNextMove]
          score += (1 - p_right)
          if p_right == maximum(p_dists)
              numRight += 1
          end
          scoref = open(scorePath, "a")
          writedlm(scoref, p_dists)
          close(scoref)
      end
      println("Score:", score)
      println("Num Right:", numRight)
      println("Num Predicitons:", nrows)
      println("Num NaN:", numNaN)
      scoref = open("$basepath$i/$scorename", "a")
      #writedlm(scoref, score)
      #writedlm(scoref, numNaN)
      close(scoref)
      println("Done scoring BN for CV #: $i")
    end #end over testnums
    println("Done with intersection", inter)
end  #end over intersections


