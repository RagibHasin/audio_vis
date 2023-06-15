#set page("a5")

= Design

== Conventions
- Two channels of audio are $x_L[n]$ and $x_R[n]$. Commonly without suffix $x[n]$.
- Sampling rate is $f_s$, thus sampling interval $ tau_s = 1 / f_s $

== Common definitions
- Interesting constant, $zeta = W((log 2)/5)) approx 0.122630300853342312$

=== Frequencies
- Maximum audio frequency, $f_max = 20 "kHz"$
- Sweet frequency, $
  cal(f)
  &= f_max zeta / (log 2) \
  &approx 3538.36253807677143 "Hz" $
- Sweet timespan, $
  cal(t)
  &= 50 mu"day"
  &= 4.32 "s" $ 
- Downsample rate, $N = 1200$

=== Viewport and positioning
- Viewport width, $w_"vp"$ #strike("= 4192 dip") = 3840 dip
- Viewport diagonal, $d = 5/4 w_"vp" = 4800 "dip"$
- Viewport aspect ratio, $rho$
- Viewport height, $h_"vp" = w_"vp" / rho = 2160 "dip"$
- Viewport area, $A_"vp" = w_"vp"^2/rho$
- Channel area, $ A_"ch"
&= h_"vp" (w_"vp" - h_"vp") + 1/2 h_"vp" (h_"vp" - 1) \ 
&= 1/2 h_"vp" (2w_"vp" - h_"vp" - 1) \
&= 5960520 "px"
$
- Initial position, $ p[0]
&= h_"vp" (w_"vp" - h_"vp") - 1/2 h_"vp" \
&= 1/2 h_"vp" (2w_"vp" - 2h_"vp" - 1) \
&= 3627720 "px"
$

=== Miscelleneous
- $gamma_C(x) = a_0 + a_1 log_2(a_2 x + a_3)$
  , where,
  $ a_0 &= 1/4 \
    a_1 &= (log 2) / (2 log((log 2) / zeta - 1)) \
        &approx 0.225432981868225421 \
    a_2 &= ((log 2 - 2 zeta) log 2) / (zeta^(3/2) sqrt(log 2 - zeta)) \
        &approx 9.57111578549689866 \
    a_3 &= sqrt(zeta / (log 2 - zeta)) \
        &approx 0.463622652910641416 $
  Derived with $gamma_C(0) = 0$, $gamma_C(cal(f) / f_max) = 1/2$ and $gamma_C(1) = 1$

== Phase ($phi$) calculation
+ Hilbert transform, $bold(z)[n]$, of signal $x[n]$ is computed.
+ Argument, $phi[n]$, of that Hilbert transform $bold(z)[n]$ is computed.
+ Argument, $phi[n]$, is then scaled and saved in a companion file e.g. `*.phases.ext`

== For image generation:
- Angular frequency, $ omega[n] = phi[n] - phi[n-1] $
  normalized, where $phi[-n] = 0$
  - $phi[n] in (-pi,pi]$
  - $omega[n] in [0,2pi)$ 
- Linear frequency, $ f[n] = f_s omega[n] / (2pi) $
- Velocity, $ v[n] = x[n] - x[n-1] $
- Decay time, $ tau_"decay"[n] = cal(t) dot 2 ^ (- |f[n]|/f_max) $
  
=== Layer, for $(n-k)^"th"$ sample
- 
- Spread coefficient, $ ς = (k tau_s) / (2 tau_"decay") $
- Spread radius, $ r_"spread" = d / 2 ς $

- Position, $ p[n] = p[n-1] + v cases(
          &p[n-1] &"if" v <= 0,
  A_"ch" -&p[n-1] &"if" v > 0,
) $

- Color: LCh-uv with $alpha$
- $h = phi + phi_0$
- $C = x sec phi$
- $L = gamma_C(|f| / f_max)$
- $alpha = gamma_C(ς)$
