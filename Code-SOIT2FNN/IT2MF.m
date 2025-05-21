function [miu_up,miu_lo] = IT2MF(c_lo, c_up,sigma,x,I)
miu_up = [];
miu_lo = [];
for i = 1: I
  [miu_up_tmp,miu_lo_tmp] = MF(c_lo(i), c_up(i),sigma(i),x(i));
  miu_up = [miu_up, miu_up_tmp];
  miu_lo = [miu_lo, miu_lo_tmp];
end
end

function [miu_upp,miu_loo] = MF(c_loo, c_upp,sigmaa,xx)

  if sigmaa < 0
      error('sigmaa is a negative value.');
  elseif sigmaa == 0
      error('sigmaa is equal to 0.');
  end

  %% Computing upper MF
  if xx < c_loo
      miu_upp = exp(-0.5*((xx - c_loo)/sigmaa)^2);
  elseif (c_loo <= xx) && (xx <= c_upp)
      miu_upp = 1;
  else%if xx > c_upp      
      miu_upp = exp(-0.5*((xx - c_upp)/sigmaa)^2);
  end
  %% Computing lower MF
  if xx <= (c_loo+c_upp)/2
      miu_loo = exp(-0.5*((xx - c_upp)/sigmaa)^2);
  else%if xx > (c_loo+c_upp)/2
      miu_loo = exp(-0.5*((xx - c_loo)/sigmaa)^2);
  end
end