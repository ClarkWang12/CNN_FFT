function [er, bad] = cnnsparsityexamples(net, x, y)
    %  feedforward
    net = cnnffsparsity(net, x);
    r = net.fv'*net.dict;
    [~, h] = max(r');
    [~, a] = max(y);
    bad = find(h ~= a);

    er = numel(bad) / size(y, 2);
end
