function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Predict user rating of each movie
movie_ratings = X * Theta';

% Compute cost
J = sum((movie_ratings(R == 1) - Y(R == 1)) .^ 2) / 2 ...
    + lambda / 2 * (sum(X.^2, 'all') + sum(Theta.^2, 'all'));

% Compute gradients for X and Theta
for movie = 1:size(X)
    users_reviewing_movie = R(movie,:) == 1;
    X_grad(movie,:) = (movie_ratings(movie, users_reviewing_movie) ...
                      - Y(movie, users_reviewing_movie)) * Theta(users_reviewing_movie, :) ...
                      + lambda * X(movie,:);
end

for user = 1:size(Theta)
    movies_reviewed_by_user = R(:, user) == 1;
    Theta_grad(user,:) = (movie_ratings(movies_reviewed_by_user, user) ...
                         - Y(movies_reviewed_by_user, user))' * X(movies_reviewed_by_user, :) ...
                         + lambda * Theta(user,:);
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
