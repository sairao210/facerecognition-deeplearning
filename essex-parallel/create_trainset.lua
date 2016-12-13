require 'paths';
require 'image';
dirpath = '/home/narshima/Documents/experiments/exp5/essex50'
--d = torch.FloatTensor()
d = {}
d['data'] = torch.FloatTensor(400,3,224,224)
d['class'] = torch.FloatTensor(400)

dt = {}
dt['data'] = torch.FloatTensor(200,3,224,224)
dt['class'] = torch.FloatTensor(200)

i = 1
j=1
for f in paths.files(dirpath) do
    print(f)	
    c = tonumber(f)
if(c) then 
    z = 0
    g = 0
    for im in paths.files(dirpath..'/'..f,'.jpg') do
	g = g+1
	if(g<=8) then
	img = image.load(dirpath..'/'..f..'/'..im,3,'float')
	ik = image.scale(img,224,224,'bilinear')
	d.data[i] = ik
        d.class[i] = c
	i = i+1
	end
	if(g<=12 and g>8) then
	img = image.load(dirpath..'/'..f..'/'..im,3,'float')
	ik = image.scale(img,224,224,'bilinear')
	dt.data[j] = ik
        dt.class[j] = c
	j = j+1
	end


--         print(im,c)
--         img = image.load(dirpath..'/'..f..'/'..im,3,'float')
--         itorch.image(img)
    end
end

--     print('----------------------------')
end
print(i)
print(j)

torch.save('trainset.t7',d)
torch.save('testset.t7',dt)
--print(d)
