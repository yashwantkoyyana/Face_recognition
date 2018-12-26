%-------------- By Yashwant Koyyana  -------------

%---------- finding number of training images in the data path specified as argument  ----------

 srcFiles = dir('path-of-the-training_images\*.png');  


% ----------------This means we reshape all 2D images of the training database----------------------
%      into 1D column vectors. Then, we combine all the 1D column vectors to construct 2D matrix 'X'.

%    path-of-the-training_images  -    path of the images used for training
%              		    X     -    A 2-Dimensional matrix, containing all 1D training image vectors.

% Make sure the Size of all the training images are the same to reduce the complexity. 


for i = 1 : length(srcFiles)

    filename = strcat('path-of-the-training_images\',srcFiles(i).name);
    img = imread(filename);
    img = rgb2gray(img);
    r=size(img,1);
    c=size(img,2);
    temp = reshape(img',r*c,1); 
    X = [X temp];       

% rgb2gray-- we first convert the image to grayscale because a color image has 3 channels , it is easier to work with one channel that 3. Plus face recognition by this method does not depend on the color of the image.
% reshape-- it takes the matrix coulmn by coulmn but we need the all pixels of the image row wise in one coulmn. Hence we are transposing the image.  
%------------------------------------------------------------------------------------------------------------------------------------------------------


%----------------Finding the mean of each row-------------------
m = mean(X,2);

%----------------The total number of training images are the number of coulmns in the X matix.-----------------
imgcount = size(X,2);

%--------------Subtract the mean vector with the face matrix to get normalized face matrix.----------
for i=1 : imgcount
    temp = double(X(:,i)) - m;
    A = [A temp];
end

%---------------Finding the co-variance matrix of A------------------------------------

L=A * A';

% if we calculate eigenvalues & eigenvectors of A*A' , then it will be very time consuming as well as memory, because of its number of rows and coulmns. 
% so we calculate eigenvalues & eigenvectors of A'*A , whose eigenvectors will be linearly related to eigenvectors of C.

L= A' * A;

%------V : eigenvector matrix------  D : eigenvalue matrix------------

[V,D]=eig(L);


%---------we can choose how many Principal Components (eigenvectors) to be taken.
%--------- for example: if corresponding eigenvalue is greater than 1, then the eigenvector will be chosen for creating eigenface

pc_eigvec = [];
for i = 1 : size(V,2) 
    if( D(i,i) > 1 )
       pc_eigvec = [pc_eigvec V(:,i)];
        
    end
end

%--- finally the eigenfaces ---------


eigenfaces = A * pc_eigvec;



%--------In this part of recognition, we compare faces by projecting the images into facespace and measuring the Euclidean distance between them.
%
%            recogimg           -   the recognized image name
%            test_image          -   the path of test image

%-----------finding the projection of each training image vector on the facespace  ----------


projectedimg = [ ]; 
for i = 1 : size(eigenfaces,2)
    temp = eigenfaces' * A(:,i);
    projectedimg = [projectedimg temp];
end



%------ extractiing PCA features of the test image---------


testimage = imread('test_image');
testimage = testimage(:,:,1);
r = size(test_image,1);
c=size(test_image,2);
temp = reshape(test_image',r*c,1);	% creating column image vector from the 2D image
temp = double(temp)-m;  		% subtracting the mean form the vector
projtestimg = eigenfaces'*temp; 	% projection of test image onto the facespace






%------calculating & comparing the euclidian distance of all projected trained images from the projected test image-------


eu_dist = [ ];

for i=1 : size(eigenfaces,2)
    temp = (norm(projtestimg-projectedimg(:,i)))^2;
    eu_dist = [eu_dist temp];
end

[eu_dist_min, index] = min(eu_dist);

recognized_img = strcat(int2str(index),'.jpg');

disp('The recognised image is');
disp(index);



------------------------------------Thanks----------------------------------














