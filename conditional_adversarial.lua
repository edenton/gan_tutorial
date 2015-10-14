local adv = {}

local bs_half = math.floor(opt.batchSize/2)
local x = torch.Tensor(opt.batchSize, unpack(opt.geometry))
local y = torch.Tensor(opt.batchSize, opt.condDim)
local targets_real = torch.Tensor(opt.batchSize)
targets_real:sub(1,bs_half):fill(1)
targets_real:sub(bs_half+1,opt.batchSize):fill(0)
local targets_fake = torch.ones(opt.batchSize)
local z = torch.Tensor(opt.batchSize, opt.noiseDim)
local ones = torch.ones(opt.batchSize)

local function updateConfusion(outputs, targets)
  for i = 1,opt.batchSize do
  local c
  if outputs[i][1] > 0.5 then c = 2 else c = 1 end
    confusion:add(c, targets[i]+1)
  end
end

local function updateD(dataset)
  grads.D:zero()

  -- first half real data
  dataset:getBatch(x, y)

  -- second half generated
  z:uniform(-1, 1)
  local x_gen = model_G:forward({z, y})
  x:sub(bs_half+1,opt.batchSize):copy(x_gen:sub(bs_half+1,opt.batchSize))

  -- forward pass
  local outputs = model_D:forward({x, y})
  local f = criterion:forward(outputs, targets_real)
  updateConfusion(outputs, targets_real)

  -- backward pass 
  local df_do = criterion:backward(outputs, targets_real)
  model_D:backward({x, y}, df_do)

  -- take gradient step
  optim.sgd(function() return 0, grads.D end, params.D, configs.D)
end

local function updateG(dataset)
  grads.D:zero()
  grads.G:zero()

  -- sample from G
  z:uniform(-1, 1)
  dataset:getBatch(nil, y)
  local x_gen = model_G:forward({z, y})
  x:copy(x_gen)

  -- forward pass through D
  local outputs = model_D:forward({x, y})
  local f = criterion:forward(outputs, targets_fake)

  -- backward pass through D and G
  local df_do = criterion:backward(outputs, targets_fake)
  local df_dx = model_D:backward({x, y}, df_do)[1]:clone()
  model_G:backward({z, y}, df_dx)

  -- take gradient step
  optim.sgd(function() return 0, grads.G end, params.G, configs.G)
end

function adv.train(dataset)
  epoch = epoch or 1
  print('\n\n[Epoch ' .. epoch .. '] learningRate = ' .. configs.G.learningRate .. ', momentum = ' .. configs.G.momentum) 
  print('<trainer> on training set:')
  for i=1,dataset:size(),opt.batchSize do
    xlua.progress(i,dataset:size())
    for k = 1,opt.K do
      updateD(dataset)
    end
    updateG(dataset)
  end
  epoch = epoch+1
  print(confusion)
  trainLogger:add{['Disriminator mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()
end

function adv.test(dataset)
  print('<trainer> on testing set:')
  for i=1,dataset:size(),opt.batchSize do
    xlua.progress(i, dataset:size())

    -- first half real data
    dataset:getBatch(x, y)

    -- second half generated
    z:uniform(-1, 1)
    local x_gen = model_G:forward({z, y})
    x:sub(bs_half+1,opt.batchSize):copy(x_gen:sub(bs_half+1,opt.batchSize))

    -- forward pass
    local outputs = model_D:forward({x, y})
    local f = criterion:forward(outputs, targets_real)
    updateConfusion(outputs, targets_real)
  end
  print(confusion)
  testLogger:add{['Disriminator mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()
end

function adv.plotSamples(N)
  local N = N or 100
  local perclass = math.floor(math.sqrt(N))
  local class = 1
  local y = torch.zeros(N, opt.condDim)
  for i = 1,100 do
    y[i][class] = 1
    if i % perclass == 0 then class = class + 1 end
  end
  local z = torch.Tensor(N, opt.noiseDim):uniform(-1, 1)
  local x_gen = model_G:forward({z, y}):clone()
  local to_plot = {}
  for n = 1,N do
    to_plot[n] = x_gen[n]:float()
  end
  torch.setdefaulttensortype('torch.FloatTensor') -- hack because image requires floats..
  local fname = paths.concat(opt.save, 'samples/' .. (epoch-1) .. '.png')

  image.save(fname,image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=math.sqrt(N)})
  torch.setdefaulttensortype(defaulttype) 
end

return adv



