require 'image'
require 'nn'
require 'torch'
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')
net = torch.load('Par-LeNetCNN(52x52-50class).t7')
net = net:cuda()
print(net:type())
trainset = torch.load('patched-data-train(52x52).t7')
print(trainset.data:type())
trainset.data = trainset.data:cuda()
trainset.class = trainset.class:cuda()
--image.display(x.data[520])
--print(#trainset.data)


-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.class[i]} 
                end}
);
--trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end



criterion = nn.ClassNLLCriterion()
--criterion = nn.CrossEntropyCriterion()
criterion = criterion:cuda()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 300 

trainer:train(trainset)

torch.save('new-net.t7',net)
