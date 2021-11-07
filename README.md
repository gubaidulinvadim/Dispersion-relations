# Stability_diagrams

Dispersion relation solver for Landau Damping.
Solving transverse and longitudinal dispersion relations written by [Scott Berg](https://doi.org/10.1109/PAC.1997.750810).
This solver is extended to deal with vlasov formalism for higher order chromaticity, RFQ and Pulsed electron lens. This was initially outlined by [M.Schenk and A.Maillard](https://doi.org/10.1103/PhysRevAccelBeams.21.084402)

## Tune spread for LHC octupoles

| Coefficient        | LHC octupoles        | FCC octupoles        |
| ------------------ | -------------------- | -------------------- |
| $a_{xx}\epsilon_n$ | $9.20 \cdot 10^{-5}$ | $1.59 \cdot 10^{-6}$ |
| $a_{xy}\epsilon_n$ | $6.54 \cdot 10^{-5}$ | $1.13 \cdot 10^{-6}$ |
| $a_{yx}\epsilon_n$ | $6.54 \cdot 10^{-5}$ | $1.66 \cdot 10^{-6}$ |
| $a_{yy}\epsilon_n$ | $9.63 \cdot 10^{-5}$ | $1.13 \cdot 10^{-6}$ |

$$\Delta Q_x = a_{xx}J_x - a_{xy}J_y$$
$$\Delta Q_y = -a_{yx}J_x + a_{yy}J_y$$

<figure>
    <img src='Results/LHC_OCT.pdf' alt="GS DC electron lens tune spread">
    <figcaption>Tune spread for octupoles (LHC)<figcaption>
<figure>

## Tune spread from an electron lens

$$\Delta Q = \Delta Q_\mathrm{MAX}\int_0^1\frac{(I_0(K_x u)-I_1(K_x u))I_0(K_y u)}}{\exp(-(K_x+K_y)u}du$$

<figure>
    <img src='Results/GS-elens.pdf' alt="LHC octupoles tune spread">
    <figcaption>Tune spread for octupoles (LHC)<figcaption>
<figure>

## Tune spread from a pulsed electron lens

$$\Delta Q_i^{x, y} = \Delta Q^{x, y}_{max}\exp\left(-\left(\frac{\tilde{\sigma_z}}{2\sigma_z}\right)^2\right)I_0\left(\left(\frac{\tilde{\sigma_z}}{2\sigma_z}\right)^2\right)$$

<figure>
    <img src='Results/PEL.pdf' alt="Pulsed KV lens tune spread">
    <figcaption>Tune spread for octupoles (LHC)<figcaption>
<figure>

## Tune spread from Radio Frequency Quadrupole(RFQ):

$$\Delta Q_i^{x, y} = \pm\Delta Q^\mathrm{RFQ}_\mathrm{MAX}J_0\left(\dfrac{\omega}{\beta c}\sqrt{2J_z\beta_z}\right)$$

<figure>
    <img src='Results/LHC_RFQ.pdf' alt="LHC RFQ(M. Schenk) tune spread">
    <figcaption>Tune spread for octupoles (LHC)<figcaption>
<figure>
