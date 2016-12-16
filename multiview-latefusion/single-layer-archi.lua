require 'image' ;
require 'nn';
require 'optim';
require 'gnuplot';

torch.setdefaulttensortype('torch.FloatTensor')
net=nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) 
net:add(nn.ReLU())                       
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())         
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolution(16, 28, 5, 5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(28*3*3))
net:add(nn.Copy(nil,nil,true))
net:add(nn.Linear(252,50))
net:add(nn.SoftMax())

torch.save('./single-layer(52x52)-for minibatch.t7',net)