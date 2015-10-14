require 'torch'
require 'nn'
pcall(function() 
  require 'cutorch' 
  require 'cunn'
end)
require 'optim'
require 'image'
require 'mnist'
require 'paths'


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -r,--learningRate  (default 0.1)         learning rate, for SGD only
  -b,--batchSize     (default 100)         batch size
  -m,--momentum      (default 0.5)           momentum
  -g,--gpu           (default -1)          on gpu 
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  --save             (default "logs")      directory to save to
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --numhidD          (default 240)         number of hidden units in D
  --numhidG          (default 1600)        number of hidden units in G
]]

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

-- fix seed
torch.manualSeed(1)


if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end
defaulttype = torch.getdefaulttensortype()

classes = {'0','1'}
opt.geometry = {1, 32, 32}

adversarial = require 'adversarial'

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

----------------------------------------------------------------------
-- define Discriminator
local numhid = opt.numhidD
model_D = nn.Sequential()
model_D:add(nn.Reshape(input_sz))
model_D:add(nn.Linear(input_sz, numhid))
model_D:add(nn.ReLU())
model_D:add(nn.Dropout())
model_D:add(nn.Linear(numhid, numhid))
model_D:add(nn.ReLU())
model_D:add(nn.Dropout())
model_D:add(nn.Linear(numhid,1))
model_D:add(nn.Sigmoid())

----------------------------------------------------------------------
-- define Generator
local numhid = opt.numhidG
model_G = nn.Sequential()
model_G:add(nn.Linear(opt.noiseDim, numhid))
model_G:add(nn.ReLU())
model_G:add(nn.Linear(numhid, numhid))
model_G:add(nn.ReLU())
model_G:add(nn.Linear(numhid, input_sz))
model_G:add(nn.Sigmoid())
model_G:add(nn.Reshape(opt.geometry[1], opt.geometry[2], opt.geometry[3]))

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

function setWeights(module, std)
  weight = module.weight
  if weight then
    weight:randn(weight:size()):mul(std)
  end
  bias = module.bias
  if bias then
    bias:zero()
  end
end

function init_model(model, std)
  for _, m in pairs(model:listModules()) do
    setWeights(m, std)
  end
end

init_model(model_D, 0.005)
init_model(model_G, 0.05)

-- retrieve parameters and gradients
params_D,grads_D = model_D:getParameters()
params_G,grads_G = model_G:getParameters()
grads = {D = grads_D, G = grads_G}
params = {D = params_D, G = params_G}

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)


----------------------------------------------------------------------
-- setup data
local ntrain = 50000
local nval = 5000

-- create training set and normalize
trainData = mnist.loadTrainSet(1, ntrain)
trainData:normalize()

-- create validation set and normalize
valData = mnist.loadTrainSet(ntrain+1, ntrain+nval)
valData:normalize()

confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_G:cuda()
end

names = {"D", "G"}
configs = {}
for _, c in pairs(names) do
  configs[c] = {}
  configs[c].learningRate = opt.learningRate
  configs[c].momentum = opt.momentum
end

os.execute('mkdir -p ' .. paths.concat(opt.save, 'samples'))

-- training loop
while true do
  -- train/test
  adversarial.train(trainData)
  adversarial.test(valData)
  adversarial.plotSamples()

  torch.setdefaulttensortype('torch.FloatTensor')
  -- plot accuracy of D
  trainLogger:style{['Disriminator mean class accuracy (train set)'] = '-'}
  testLogger:style{['Disriminator mean class accuracy (test set)'] = '-'}
  trainLogger:plot()
  testLogger:plot()
  torch.setdefaulttensortype(defaulttype)

  for _, c in pairs(names) do
    configs[c].learningRate = math.max(configs[c].learningRate / 1.0004, 0.000001)
    configs[c].momentum = math.min(configs[c].momentum + 0.002, 0.7)
  end 

end
