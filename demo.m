%%This is a demo for removing outliers. In this demo, the SIFT matches  is known in advance.
clear;
close all; 

%class = 1; method = 'affine'
%class = 2; method = 'rigid';
class = 3; method = 'nonrigid';
k = 1;

conf.lambda = 9 * (10^9);
conf.Kn = 15;

%affine
al = {'2-refimg(2).tif',  '6-refimg(6).tif'};
ar = {'2-t.bmp', '6-t.bmp'};

%rigid
rl = {'batch_2-2.tif','Sh2000.tif'};
rr = {'batch_2-3.tif','Sh2001.tif'};

%nonrigid
nl = {'C23.bmp', '621.bmp'};
nr = {'C13.bmp', '611.bmp'};

switch class
    case 1
        fn_l = al;fn_r = ar;
    case 2
        fn_l = rl;fn_r = rr;
    case 3
        fn_l = nl;fn_r = nr;
end
k = min(k, numel(fn_l));
Ia = imread(fn_l{k});
Ib = imread(fn_r{k});
if size(Ia,3)==1
    Ia = repmat(Ia,[1,1,3]);
end
if size(Ib,3)==1
    Ib = repmat(Ib,[1,1,3]);
end

%%
fn1 = fn_l{k};
k1 = strfind(fn1, '.');
k1 = k1(end);
fn2 = fn_r{k};
k2 = strfind(fn2, '.');
k2 = k2(end);
tmp = [fn1(1:k1-1) '_' fn2(1:k2-1)];
fn_match = ['./data/' tmp '.mat'];
flag = 0;
if ~exist(fn_match, 'file') || flag
   [X, Y] = sift_match(Ia, Ib, SiftThreshold, fn_match);
else
    load(fn_match);
end
tmp = [tmp '_' method];
[nX, nY, normal]=norm2(X,Y);
if ~exist('conf'), conf = []; end
conf = LLT_init(conf);
switch method
    case 'affine'
        VecFld=LLTA(nX, nY, conf);
    case 'rigid'
        VecFld=LLTR(nX, nY, conf);
    case 'nonrigid'
        VecFld=LLTV(nX, nY, conf);
end
VecFld.TX=(VecFld.TX)*normal.yscale+repmat(normal.ym,size(Y,1),1);
[precise, recall, corrRate] = evaluate(CorrectIndex, VecFld.VFCIndex, size(X,1));
[wa,ha,~] = size(Ia);[wb,hb,~] = size(Ib);maxw = max(wa,wb);maxh = max(ha,hb);Ib(wb+1:maxw, :,:) = 0;Ia(wa+1:maxw, :,:) = 0;
plot_matches(Ia, Ib, X, Y, 1:size(X,1), CorrectIndex);
figure;
plot_matches(Ia, Ib, X, Y, VecFld.VFCIndex, CorrectIndex);




  
