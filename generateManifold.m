%generateManifold - create a set of points on a manifold 
%
% generateManifold({PropertyList}) creates a file filled with points
% sampled from a low dimensional mainfold in a higher dimensional
% space.  These points
% are normalized to exist between -1 and 1 in all dimensions.
%
% Recognized Properties
%
% EmbeddedDimensions - number of dimensions in the embedding space (M)
% ManifoldDimensions - Dimension of the manifold  (N)
% FileName - output file name
% NumPoints - the number of points generated
%
% Approach
% 
% The approach here is to move a line through the embedding space and,
% after each step, randomly place points inside an M dimensional
% ellipsoid defined by (M-1) axes orthogonal to the velocity
% direction. These points are gaussian around the location along the
% line with variances along each fo the (M-1) orthogonal axes fixed.
%
% At a give time, the state of the system is represented by a set of N
% M-dimensional vectors. The first of these is the velocity vector,
% the rest are used to determine the orientation of the ellipsoid in M
% dimensional space. The lengths of these vectors determines the size
% of the ellipsoid along their embedded space.
%
% With each update, the origin is offset by a random amount along the
% first ellipsoid axis. The offset along this axis is chi2 with 1
% degree of freedom.
%
% All of the axes are then randomly offset by small amount by adding a
% gaussian offset to their endpoints then renormalizing them and
% re-orthognalizing them. The offsetes to the first axes will give the
% path of the line a wandering path in the embedding space. The rest
% will give a small rotation around the axes.
function generateManifold(varargin)

Properties.NumSteps = 16*1024; 
Properties.NumPoints = 16; 
Properties.EmbeddingDimensions = 2;
Properties.ManifoldDimensions = 2;
Properties.ManifoldExtents = [];
Properties.FileName = 'Manifold.dat';
Properties.Scatter = 0.1;

Properties = setProperties(Properties,varargin{:});

if (isempty(Properties.ManifoldExtents))
    Properties.ManifoldExtents = 0.001*ones(Properties.ManifoldDimensions,1);
else
    Properties.ManifoldExtents = Properties.ManifoldExtents(:);
end

if (isscalar(Properties.ManifoldExtents))
    Properties.ManifoldExtents = ...
        Properties.ManifoldExtents * ones(Properties.ManifoldDimensions,1);
end

if (length(Properties.ManifoldExtents) ~= Properties.ManifoldDimensions)
    error ('Inconsistent Manifold Extents');
end

% Set up some shorter varialbe names
N = Properties.ManifoldDimensions;
M = Properties.EmbeddingDimensions;
Extents = Properties.ManifoldExtents;
NumPoints = Properties.NumPoints;

Location = zeros(M,1);

% Create the orientation vectors so that they are orthonormal
Orientation = zeros(M,N);
Orientation(1:N,1:N) = eye(N);

% We start at the origin with the velcoty along the first dimension
Points = zeros(M,NumPoints * Properties.NumSteps);
PointIndex = 1;
for Index = 1:Properties.NumSteps
    
    % To do the point generation, for every random point, we are
    % going to have an offset along each of the directions of the
    % manifold. So we need a NumPoints x N x M tensor. 
    % This matrix is the extent along each of the manifold directions that
    % we are going to offset from the location. 
    for I = 1:NumPoints
        Temp = randn(1,N) .* Extents(:)';
        Points(:,PointIndex) = Location + sum(Orientation .* Temp,2);
        PointIndex = PointIndex+1;
    end
    
    % Now, let's move the location
    Location = Location + 0.5*Extents(1)*Orientation(:,1) * chi2rnd(1);
    
    % Now let's offset all the axes for a random rotation
    Orientation = Orientation + Properties.Scatter * randn(M,N);
    
    % Now let's normalize them all
    Orientation = Orientation./sqrt(sum(Orientation.^2,1));
    
    % Now orthogonalize them all. This is done with a graham-schmidt reduction
    for I = 2:N
        
        % Get the subspace of the prior axes
        SubSpace = Orientation(:,1:(I-1));

        % Find the projections of this vector along those axes
        Projections = sum(SubSpace .* repmat(Orientation(:,I),1,(I-1)));

        % Subtract those projections.
        Orientation(:,I) = Orientation(:,I) - sum(Projections .*SubSpace,2);
        
        % Renormalize
        Orientation(:,I) = Orientation(:,I) / norm(Orientation(:,I));
    end

end

FID = fopen(Properties.FileName,'w');
Format = [repmat('%.8e ',1,M) '\n'];
fprintf(FID,Format,Points(:));
fclose(FID);

