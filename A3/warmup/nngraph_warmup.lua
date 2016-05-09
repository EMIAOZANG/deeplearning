
require 'nn';
require 'nngraph';

---------------
-- 1(a)
---------------

xx = nn.Identity()()
yy = nn.Identity()()
zz = nn.Identity()()

--Linear transformation
wx_b = nn.Linear(4,2)({xx})
wy_b = nn.Linear(5,2)({yy})

--Non-Linear transformation
tanh = nn.Tanh()({wx_b})
sig = nn.Sigmoid()({wy_b})

--Squaring
sq_tanh = nn.Square()({tanh})
sq_sig = nn.Square()({sig})

--Pairwise multiplication
cproduct = nn.CMulTable()({sq_tanh, sq_sig})

--Adding z
a = nn.CAddTable()({cproduct, zz})

--assemble the graph
model = nn.gModule({xx, yy, zz},{a})

-----------------
-- 1(b)
-----------------

--Choose values for x, y, z and gradOutput
x = torch.ones(4)
y = torch.ones(5)
z = torch.ones(2)
gradOutput = torch.ones(2)


--Prints forward pass and backward pass
function fp_bp(model_, x_, y_, z_, grad)
    print("Forward prop outout: ", model_:forward({x_, y_, z_}))
    print("Backward prop output: ", model_:backward({x_, y_, z_}, grad))
end

--Run
fp_bp(model, x, y, z, gradOutput)


