require 'image'
require 'nn'
require 'torch'
require 'optim'
torch.setdefaulttensortype('torch.FloatTensor')
net = torch.load('latefusion(52x52)-for minibatch.t7')
print(net:type())
trainset = torch.load('patched-data-train(52x52).t7')
print(trainset.data:type())



Tsize = trainset.data:size(1)
print(Tsize)
shuffle = torch.randperm(Tsize)
Sdata = torch.FloatTensor(Tsize,10,3,52,52)
Slab = torch.FloatTensor(Tsize)
for i=1, Tsize do
    Sdata[i]:copy(trainset.data[shuffle[i]])
    Slab[i] = trainset.class[shuffle[i]]
end



logger = optim.Logger('losses.log')
logger:setNames{'Epoch loss'}
Bsize = 16
Tsize = trainset.data:size(1)
Nbatch = math.floor(Tsize/Bsize)
if(not(Tsize/Bsize - Nbatch == 0)) then
    Nbatch = Nbatch+1 
end
print(Nbatch)



criterion = nn.ClassNLLCriterion()
params, gradParams = net:getParameters()
local optimState = {learningRate = .01, learningRateDecay = 1e-7}
local nEpochs = 50
for epoch = 1, nEpochs do
    Tloss = 0
    for i =1,Nbatch do
        if(i == Nbatch) then
            a = (i-1)*Bsize+1;b = Tsize;
        else
            a = (i-1)*Bsize+1;b = i*Bsize; end

        local inp = torch.FloatTensor(b-a+1,10,3,52,52)
        local lab = torch.FloatTensor(b-a+1)

        inp:copy(Sdata[{{a,b},{},{},{},{}}])
        lab:copy(Slab[{{a,b}}])

       function feval(params)
          gradParams:zero()

          local outputs = net:forward(inp)
          local loss = criterion:forward(outputs, lab)
          local dloss_doutputs = criterion:backward(outputs, lab)
          net:backward(inp, dloss_doutputs)
          Bloss = loss
--           print('epoch number '..epoch.. ' batch number is ' .. i ..' loss is '..loss)
          return loss, gradParams
       end

       optim.sgd(feval, params, optimState)
        Tloss = Tloss + Bloss        
    end
    Tloss = Tloss/Nbatch
    logger:add{Tloss}
    print('epoch number '..epoch..' loss is '..Tloss)
    
end

logger:style{'+-'}
logger:plot()
