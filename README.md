The codes are for training a 3D GAN model.
The GAN architecture is designed with a custom loss function and utilization of morphometric data for improving the fidelity of geometry in the generaed designs.
I named it MDI-GAN.
This is a course Project for UIUC CEE-598 course, Fall 2025.
GAN architecture:
<img width="1431" height="747" alt="image" src="https://github.com/user-attachments/assets/518671d4-84eb-41ed-a9d1-0db0c4e68edc" />

Loss functions:
L_D^real=E_(x∼p_data ) [max⁡(0, 1-D(x)) ]
L_D^fake=E_(z∼p_z ) [max⁡(0, 1+D(G(z))) ]
L_D=L_D^real+L_D^fake
L_G=-E_(z,y) [ D(G(z,y),y) ]
L_VHR=| VHR(G(z,y))-VHR_target |
L_(MSE-height)=( h(G(z,y))-h_target )^2
L_G=-E_(z,y) [ D(G(z,y),y) ]+λ〖(L〗_VHR+L_(MSE-height))

Comparison(This model vs Vanilla GAN):

This model:
<img width="1408" height="384" alt="image" src="https://github.com/user-attachments/assets/241b2f2b-2c0a-433d-88bc-ed353dd8858d" />

Vanilla GAN:

<img width="345" height="345" alt="image" src="https://github.com/user-attachments/assets/87fc56c1-a7e8-4c66-a479-39db24c64f9c" /><img width="345" height="345" alt="image" src="https://github.com/user-attachments/assets/83b71b16-86ad-462b-ab77-82ce8d25609b" /> <img width="345" height="345" alt="image" src="https://github.com/user-attachments/assets/c9b9b56a-b309-4095-a6cf-6beab4b4a235" /> <img width="345" height="345" alt="image" src="https://github.com/user-attachments/assets/5a2049f0-0546-4d79-85f1-66fd076316e0" />

MDI-cGAN training loss:
<img width="683" height="473" alt="image" src="https://github.com/user-attachments/assets/839206d3-b374-4d69-94c6-e076f7109efb" />



