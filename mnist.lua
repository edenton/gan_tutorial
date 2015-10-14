require 'torch'
require 'paths'

mnist = {}

mnist.path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
mnist.path_dataset = 'mnist.t7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_32x32.t7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test_32x32.t7')

function mnist.download()
  if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
    local remote = mnist.path_remote
    local tar = paths.basename(remote)
    os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
  end
end


function mnist.loadTrainSet(start, stop)
  return mnist.loadDataset(mnist.path_trainset, start, stop)
end

function mnist.loadTestSet()
  return mnist.loadDataset(mnist.path_testset)
end

function mnist.loadDataset(fileName, start, stop)
  mnist.download()

  local f = torch.load(fileName, 'ascii')
  local data = f.data:type(torch.getdefaulttensortype())
  local labels = f.labels:type("torch.IntTensor")

  local nExample = f.data:size(1)
  local start = start or 1
  local stop = stop or nExample
  if stop > nExample then
    stop = nExample
  end 
  local labels = labels[{{start, stop}}]
  local data = data[{{start, stop}}]
  local N = stop - start + 1 

  print('<mnist> loaded ' .. N .. ' examples.') 

  local dataset = {}
  dataset.data = data
  dataset.labels = labels

  function dataset:size()
    return N 
  end 

  function dataset:normalize()
    local new_min = 0
    local new_max = 1
    local old_max = self.data:max(1)
    local old_min = self.data:min(1)
    local eps = 1e-7
    for i=1,N do
      self.data[i]:add(-1, old_min)
      self.data[i]:mul(new_max - new_min)
      self.data[i]:cdiv(old_max - old_min + eps)
      self.data[i]:add(new_min)
    end 
  end

  function dataset:getBatch(x, label)
    local n
    if x then n = x:size(1) else n = label:size(1) end
    for i = 1,n do
      local idx = math.random(N)
      if x then
        x[i]:copy(self.data[idx])
      end
      if label then
        label[i]:zero()
        label[i][self.labels[idx]] = 1
      end 
    end 
    return x
  end 

  return dataset
end
