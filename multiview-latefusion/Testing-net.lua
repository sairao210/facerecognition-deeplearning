require 'nn';
require 'paths';
require 'image';
require 'csvigo';
net = torch.load('./new-net.t7')

testdata = torch.load('./patched-data-test(52x52).t7')
mean = {0.34332438465718,0.27035354647333,0.28584258351481}
stv = {0.20793748948497,0.1577492339104,0.18378318819282}
for i=1,3 do
    testdata.data[{ {}, {}, {i}, {}, {}  }]:add(-mean[i]) 
    testdata.data[{ {}, {}, {i}, {}, {}  }]:div(stdv[i]) 
end
Tsize = testdata.data:size(1)
prediction = net:forward(testdata.data)
correct = 0
for i=1,Tsize do
    print(i)
    local groundtruth = testdata.class[i]
--     local prediction = net:forward(testdata.data[i])
    local confidences, indices = torch.sort(prediction[i], true)
    if groundtruth == indices[1] then
    print('Ha Ha')        
    correct = correct + 1
    end
end
print(correct, 100*(correct/Tsize) .. ' % ')
