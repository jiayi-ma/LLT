function [precision, recall, corrRate] = evaluate(CorrectIndex, VFCIndex, siz,fid)
%   [PRECISION, RECALL, CORRRATE] = EVALUATE(CORRECTINDEX, VFCINDEX, SIZ)
%   evaluates the performence of VFC with precision and recall.
%
% Input:
%   CorrectIndex, VFCIndex: Correct indexes and indexes reserved by VFC.
%
%   siz: Number of initial matches.
%
% Output:
%   precision, recall, corrRate: Precision and recall of VFC, percentage of
%       initial correct matches.
%
%   See also:: VFC().

if length(VFCIndex)==0
    VFCIndex = 1:siz;
end

tmp=zeros(1, siz);
tmp(VFCIndex) = 1;
tmp(CorrectIndex) = tmp(CorrectIndex)+1;
VFCCorrect = find(tmp == 2);
NumCorrectIndex = length(CorrectIndex);
NumVFCIndex = length(VFCIndex);
NumVFCCorrect = length(VFCCorrect);

corrRate = NumCorrectIndex/siz;
precision = NumVFCCorrect/NumVFCIndex;
recall = NumVFCCorrect/NumCorrectIndex;

if nargin < 4
    fprintf('\ncorrect correspondence rate in the original data: %d/%d = %f\r\n', NumCorrectIndex, siz, corrRate);
    fprintf('precision rate: %d/%d = %f\r\n', NumVFCCorrect, NumVFCIndex, precision);
    fprintf('recall rate: %d/%d = %f\r\n', NumVFCCorrect, NumCorrectIndex, recall);
else
    fprintf(fid,'\ncorrect correspondence rate in the original data: %d/%d = %f\r\n', NumCorrectIndex, siz, corrRate);
    fprintf(fid,'precision rate: %d/%d = %f\r\n', NumVFCCorrect, NumVFCIndex, precision);
    fprintf(fid,'recall rate: %d/%d = %f\r\n', NumVFCCorrect, NumCorrectIndex, recall);
end