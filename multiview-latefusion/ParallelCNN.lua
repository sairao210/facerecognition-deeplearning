require 'image' ;
require 'nn';
require 'optim';
require 'gnuplot';

torch.setdefaulttensortype('torch.FloatTensor')
mlp = nn.Sequential();
c = nn.Parallel(1,1)

for i=1,10 do
    local net=nn.Sequential()
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
    c:add(net)
end
mlp:add(c)
mlp:add(nn.Linear(2520,1000))
mlp:add(nn.Dropout())
mlp:add(nn.Linear(1000,400))
mlp:add(nn.Linear(400,50))
mlp:add(nn.SoftMax())

torch.save('./multiview-latefusion(52x52).t7',mlp)
