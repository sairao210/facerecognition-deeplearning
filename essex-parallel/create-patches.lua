require 'nn'
require 'image'
trainset = torch.load('trainset.t7')
print(trainset.data:size())
print(trainset.data[1]:size())
---image patch formation
pat1 = trainset.data[{ {},{},{1,187},{1,187} }]
pat2 = trainset.data[{ {},{},{37,224},{37,224} }]
pat3 = trainset.data[{ {},{},{1,187},{37,224} }]
pat4 = trainset.data[{ {},{},{37,224},{1,187} }]
pat5 = trainset.data[{ {},{},{1,157},{} }]
pat6 = trainset.data[{ {},{},{},{1,157} }]
pat7 = trainset.data[{ {},{},{67,224},{} }]
pat8 = trainset.data[{ {},{},{},{67,224} }]
pat9 = trainset.data[{ {},{},{18,206},{18,206} }]
pat10 = trainset.data[{ {},{},{},{} }]

input = torch.FloatTensor(400,10,3,52,52)
for i=1,400 do
    patch = torch.FloatTensor(10,3,52,52)
    patch[1] = image.scale(pat1[i],52,52)
    patch[2] = image.scale(pat2[i],52,52)
    patch[3] = image.scale(pat3[i],52,52)
    patch[4] = image.scale(pat4[i],52,52)
    patch[5] = image.scale(pat5[i],52,52)
    patch[6] = image.scale(pat6[i],52,52)
    patch[7] = image.scale(pat7[i],52,52)
    patch[8] = image.scale(pat8[i],52,52)
    patch[9] = image.scale(pat9[i],52,52)
    patch[10] = image.scale(pat10[i],52,52)
    input[i] = patch
end

print(input:size())

--tonumber(testdata.class[i])
for i=1,400 do
    trainset.class[i]=tonumber(trainset.class[i])
    end
data = {}
data['data'] = input
data['class'] = trainset.class

torch.save('./patched-data-train(52x52).t7',data)
