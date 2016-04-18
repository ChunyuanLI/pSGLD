function accessor = getExpsAccessor(attr)
	accessor = @f;
	function curve = f(exp)
		curve.label = exp.name;
		curve.y = exp.results.(attr);
		curve.x = 1:numel(curve.y);
        shifted = [0 ; exp.results.times(1:end-1)];
        curve.avgTime = mean(exp.results.times - shifted);
	end
end