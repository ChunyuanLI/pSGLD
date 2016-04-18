function accessor = getExpsAccessor(attr)
	accessor = @f;
	function curve = f(exp)
        alg = exp.name;
        dashPos = strfind(alg,'-');
        if ~isempty(dashPos); alg = alg(1:dashPos-1); end
        lr = exp.descentOpts.learningRate;
        if isfield(exp.descentOpts,'epsilon')
            eps = exp.descentOpts.epsilon;
            curve.label = sprintf('%s:Lr=%.e,Ep=%.e',alg,lr,eps);
        else
            curve.label = sprintf('%s:Lr=%.e',alg,lr);
        end
        curve.label = strrep(curve.label,'e-0','e-');
		curve.y = exp.results.(attr);
		curve.x = 1:numel(curve.y);
        shifted = [0 ; exp.results.times(1:end-1)];
        curve.avgTime = mean(exp.results.times - shifted);
	end
end