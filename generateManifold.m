%generateManifold - create a set of points on a manifold 
%
% generateManifold({PropertyList}) creates a file filled with points
% sampled from a low dimensional mainfold in a higher dimensional
% space. 
%
% Recognized Properties
%
% FileName - output file name
% NumSteps = the number of steps
% NumPoints - the number of points generated per step
% Velocity - speed (distance per step).
% ManifoldExtents - the length of each of the manifold dimensions
% ManifoldNoise - the randomness around each of the manifold axes
% BirthProbability - the per-step probability that a given path will split
% DeathProbability - the per-step probabiliy that a given path will die.
% EmbeddedDimensions - number of dimensions in the embedding space (M)
% ManifoldDimensions - Dimension of the manifold  (N)
%
% Approach
% 
% The approach here is to create paths by tracing lines through the
% embedding space and, after each step, randomly place points inside
% or on an M dimensional ellipsoid defined by (M-1) axes orthogonal to
% the velocity direction of the lines. These points are gaussian
% around the location along the line with variances along each of the
% (M-1) orthogonal axes fixed.
%
% At a given time, the state of each path in the system is represented
% by a set of N M-dimensional "orientation" vectors and an M
% dimensional location. The first of these is the velocity vector. The
% rest are used to determine the orientation of the ellipsoid in M
% dimensional space. The lengths of these vectors determines the size
% of the ellipsoid along their embedded space, however their lengths
% are maintained in the algorithm separately from the orientation
% direction vectors, which are unit-normal.
%
% With each update, the origin is offset by a random amount along the
% first ellipsoid axis. The offset along this axis is chi2 with 1
% degree of freedom.
%
% All of the axes are then randomly offset by small amount by adding a
% gaussian offset to their endpoints then renormalizing them and
% re-orthognalizing them. The offsets to the first axes will give the
% path of the line a wandering path in the embedding space. The rest
% will give a small rotation around the various axes.
%
% In addition, at each step there is a probability that the path
% will "split" into to and also a probbilty that a path may
% vanish. This gives the walk a tree-like structure. 
function generateManifold(varargin)

Properties.NumSteps = 16*1024; 
Properties.FileName = 'Manifold.dat';
Properties.NumPoints = 16; 
Properties.Velocity = 0.01;
Properties.ManifoldExtents = [0.01 0.1];
Properties.ManifoldNoise = 0.05;
Properties.BirthProbability = 0.00;
Properties.DeathProbability = 0.00;
Properties.ManifoldDimensions = 2;
Properties.EmbeddingDimensions = 3;

Properties = setProperties(Properties,varargin{:});

% Each of these properties needs to have the dimension of the manifold. If they are scalar, replicate them, otherwise, error
for FieldName = {'ManifoldExtents' 'ManifoldNoise'}
    FieldName = FieldName{1};
    Field = Properties.(FieldName);
    Field = Field(:);
    if (isscalar(Field))
        Field = Field * ones(Properties.ManifoldDimensions,1);
    else
        if (length(Field) ~= Properties.ManifoldDimensions)
            error 'Bad Field Dimension';
        end
    end
    Properties.(FieldName) = Field;
end

% Set up some shorter varialbe names
NumPoints = Properties.NumPoints;
N = Properties.ManifoldDimensions;
M = Properties.EmbeddingDimensions;
Extents = Properties.ManifoldExtents;

% This program iterates paths through the embedding dimension,
% carrying with it the N dimensional orientation axes that describe
% the N dimensional manifold. WE start with just one such path
Paths.Location = zeros(M,1);

% Create the orientation vectors so that they are orthonormal.
%
% This is an M by N by P tensor, where M is the embedding dimension, N
% is the manifold dimension, and P is the number of paths, which we
% set to 1 to start. We will augment this when paths are born and
% decimate it when they are killed.
Paths.Orientation = zeros(M,N);
Paths.Orientation(1:N,1:N,1) = eye(N);

% Open the output file and define the format
FID = fopen(Properties.FileName,'w');
Format = [repmat('%.8e ',1,M) '\n'];

% We start at the origin with the velcoty along the first dimension
fprintf('Start: ');
for Index = 1:Properties.NumSteps
    
    % To do the point generation, for every random point, we are
    % going to have an offset along each of the directions of the
    % manifold. So we need a NumPoints x N x M tensor. 
    % This matrix is the extent along each of the manifold directions that
    % we are going to offset from the location. 
    if (rem(Index,1024) == 1)
      fprintf('.');
    end

    for PathIndex = 1:length(Paths)
        Path = Paths(PathIndex);
        for  PointIndex = 1:NumPoints
            Temp = randn(1,N) .* Extents(:)';
            fprintf(FID,Format,Path.Location + ...
                    sum(squeeze(Path.Orientation) .* Temp,2));
        end
    end

    % We now iterate to the next step. First, we randomly birth according
    % to the given probabilities. We randomly birth from only the
    % existing paths.
    NumPaths = length(Paths);
    for PathIndex = 1:NumPaths
        Temp = rand(1);
        if (Temp < Properties.BirthProbability)
            fprintf('B');
            Path(end+1) = Paths(PathIndex);
        end
    end

    % Now kill some guys. Note that we cannot kill paths we just
    % made because of the limit 
    PathIndex = 1;
    while (NumPaths > 1 && PathIndex <= NumPaths)
        Temp = rand(1);
        if (Temp < Properties.DeathProbability)
            fprintf('D');
            Paths = Paths(setxor(1:length(Paths),PathIndex));
            NumPaths = NumPaths-1;
        else
            PathIndex = PathIndex+1;
        end
    end
    
    % Now, let's move all the paths
    for PathIndex = 1:length(Paths)
        Path = Paths(PathIndex);
        Path.Location = ...
            Path.Location + ...
            0.5*Properties.Velocity*Path.Orientation(:,1) * chi2rnd(1);
        
        % Now let's offset all the axes for a random rotation
        Orientation = Path.Orientation + Properties.ManifoldNoise' .* randn(M,N);
        
        % Now let's normalize them all
        Orientation = Orientation./sqrt(sum(Orientation.^2,1));
        
        % Now orthogonalize them all. This is done with a graham-schmidt
        % reduction
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

        Path.Orientation = Orientation;
        Paths(PathIndex) = Path;
    end
end
fprintf('\n');
fclose(FID);

