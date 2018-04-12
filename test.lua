
require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'nngraph'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'stn'


opt = {
    DATA_ROOT = './Datasets',   --DataRoot
    batchSize = 1,            -- # images in batch
    loadSize = 256,           -- scale images to this size
    fineSize = 256,           --  then crop to this size
    flip=0,                   -- horizontal mirroring data augmentation
    gpu = 1,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X (cpu untested)
    which_direction = 'AtoB', -- AtoB or BtoA
    phase = 'RealImg',            -- test dataset name
    preprocess = 'regular',   -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
    aspect_ratio = 1.0,       -- aspect ratio of result images
    name = 'FaceRestoration',                -- name of experiment, selects which model to run, should generally should be passed on command line
    input_nc = 3,             -- #  of input image channels
    output_nc = 3,            -- #  of output image channels
    serial_batches = 1,       -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,    -- iter into serial image list
    cudnn = 1,                -- set to 0 to not use cudnn (untested)
    checkpoints_dir = './checkpoints', -- loads models from here
    results_dir='./results',          -- saves results here
    which_model = 'netG',            -- which epoch to test? 

}


-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1 -- test only works with 1 thread...
print(opt)


opt.manualSeed = torch.random(1, 10000) -- set seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.netG_name = opt.name .. '/' .. opt.which_model
local data_loader = paths.dofile('data2/dataC.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
opt.how_many=data:size()
---------------------------------------------------------------------------------------------------

local input = torch.FloatTensor(opt.batchSize,6,opt.fineSize,opt.fineSize)
local output = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
local guidance = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
local outputface = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt):cuda()

netG:apply(printNet)--print network
local filepaths = {} -- paths to images tested on
local filenames ={}
function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end


for n=1,math.floor(opt.how_many/opt.batchSize) do
    print('processing batch ' .. n)
    
    local real_data, filepaths_curr, flips = data:getBatch()
    local imgname2 = filepaths_curr[1]

    filepaths_curr = util.basename_batch(filepaths_curr)
    print('filepaths_curr: ', filepaths_curr)
    
    real_A=real_data[{ {}, {1,3}, {}, {} }]:clone() -- Blur image
    real_B=real_data[{ {}, {4,6}, {}, {} }]:clone() -- guidance

    local outputss = netG:forward({real_A:cuda(),real_B:cuda()})
    real_WC=outputss[1]:clone()--warped guidance
    fake_gout=outputss[3]:clone()--restoration result

    input = util.deprocess_batch(real_A):float()
    outputwarp = util.deprocess_batch(real_WC):float()
    guidance = util.deprocess_batch(real_B):float()
    outputface = util.deprocess_batch(fake_gout):float()


    -- save images
    paths.mkdir(paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase))
    local image_dir = paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase, 'images')
    paths.mkdir(image_dir)
    paths.mkdir(paths.concat(image_dir,'Input'))
    paths.mkdir(paths.concat(image_dir,'WarpGuidance'))
    paths.mkdir(paths.concat(image_dir,'Guidance'))
    paths.mkdir(paths.concat(image_dir,'Output'))

    for i=1, opt.batchSize do
        image.save(paths.concat(image_dir,'Input',filepaths_curr[i]), image.scale(input[i],input[i]:size(2),input[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'WarpGuidance',filepaths_curr[i]), image.scale(outputwarp[i],outputwarp[i]:size(2),outputwarp[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'Guidance',filepaths_curr[i]), image.scale(guidance[i],guidance[i]:size(2),guidance[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'Output',filepaths_curr[i]), image.scale(outputface[i],outputface[i]:size(2),outputface[i]:size(3)/opt.aspect_ratio))
    end
    print('Saved images to: ', image_dir)
    BB=string.split(imgname2,'/')
    local iename = BB[#BB]
    local ImgName={}
    ImgName[1]=iename

    filepaths = TableConcat(filepaths, filepaths_curr)
    filenames = TableConcat(filenames, ImgName)
end

-- make webpage
io.output(paths.concat(opt.results_dir,opt.netG_name .. '_' .. opt.phase, 'index.html'))
io.write('<meta http-equiv="Content-Type" content="tet/html;charset=UTF-8"><table style="text-align:center;">')
io.write('<tr><td>Image #</td><td>Input</td><td>Guidance</td><td>Warped Guidance</td><td>Output</td></tr>')
for i=1, #filepaths do
    io.write('<tr>')
    io.write('<td>' .. filenames[i] .. '</td>')
    io.write('<td><img src="./images/Input/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./images/Guidance/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./images/WarpGuidance/' .. filepaths[i] .. '"/></td>')
    io.write('<td><img src="./images/Output/' .. filepaths[i] .. '"/></td>')
    io.write('</tr>')
end
io.write('</table>')
