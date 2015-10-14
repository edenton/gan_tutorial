local adv = {}

local bs_half = math.floor(opt.batchSize/2)
local x = torch.Tensor(opt.batchSize, unpack(opt.geometry))
local targets_real = torch.Tensor(opt.batchSize)
targets_real:sub(1,bs_half):fill(1)
targets_real:sub(bs_half+1,opt.batchSize):fill(0)
local targets_fake = torch.ones(opt.batchSize)
local z = torch.Tensor(opt.batchSize, opt.noiseDim)

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
  dataset:getBatch(x:sub(1,bs_half))

  -- second half generated
  z:uniform(-1, 1)
  local x_gen = model_G:forward(z)
  x:sub(bs_half+1,opt.batchSize):copy(x_gen:sub(1,bs_half))

  -- forward pass
  local outputs = model_D:forward(x)
  local f = criterion:forward(outputs, targets_real)
  updateConfusion(outputs, targets_real)

  -- backward pass 
  local df_do = criterion:backward(outputs, targets_real)
  model_D:backward(x, df_do)

  -- take gradient step
  optim.sgd(function() return 0, grads.D end, params.D, configs.D)
end

local function updateG(dataset)
  grads.D:zero()
  grads.G:zero()

  -- sample from G
  z:uniform(-1, 1)
  local x_gen = model_G:forward(z)
  x:copy(x_gen)

  -- forward pass through D
  local outputs = model_D:forward(x)
  local f = criterion:forward(outputs, targets_fake)

  -- backward pass through D and G
  local df_do = criterion:backward(outputs, targets_fake)
  local df_dx = model_D:backward(x, df_do):clone()
  model_G:backward(z, df_dx)

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
    dataset:getBatch(x:sub(1,bs_half))

    -- second half generated
    z:uniform(-1, 1)
    local x_gen = model_G:forward(z)
    x:sub(bs_half+1,opt.batchSize):copy(x_gen:sub(1,bs_half))

    -- forward pass
    local outputs = model_D:forward(x)
    local f = criterion:forward(outputs, targets_real)
    updateConfusion(outputs, targets_real)
  end
  print(confusion)
  testLogger:add{['Disriminator mean class accuracy (test set)'] = confusion.totalValid * 100}
  confusion:zero()
end


function adv.plotSamples(N)
  local N = N or 64
  local z = torch.Tensor(N, opt.noiseDim):uniform(-1, 1)
  z_fixed = z_fixed or torch.Tensor(N, opt.noiseDim):uniform(-1, 1)
  local x_gen = model_G:forward(z):clone()
  local to_plot = {}
  for n = 1,N do
    to_plot[n] = x_gen[n]:float()
  end

  local x_gen = model_G:forward(z_fixed):clone()
  local to_plot_fixed = {}
  for n = 1,N do
    to_plot_fixed[n] = x_gen[n]:float()
  end
  
  torch.setdefaulttensortype('torch.FloatTensor') -- hack because image requires floats..
  local fname = paths.concat(opt.save, 'samples/' .. (epoch-1) .. '.png')
  image.save(fname,image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=math.sqrt(N)})
  local fname = paths.concat(opt.save, 'sample.png')
  image.save(fname,image.toDisplayTensor{input=to_plot_fixed, scaleeach=true, nrow=math.sqrt(N)})
  torch.setdefaulttensortype(defaulttype) 
end

return adv



