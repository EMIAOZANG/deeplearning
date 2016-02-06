-- results.lua
--[[
args:
	model: model saved in training 
	test data: dataset

returns:
	saves a list of predicted values
]]



-- get model and test data from command line?

-- write header
file = io.open("test.csv","w")
file:write("Id,Prediction\n")
io.close(file)

-- write results
file = io.open("test.csv","a")
for key,val in pairs(predictions) do
    file:write(key .. ',' .. val .. "\n")
end
io.close(file)