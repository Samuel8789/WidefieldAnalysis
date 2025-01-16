width=1280;
height=720;
% RECTANGLE
% Define the size of the main rectangle
rect_width = 150;
rect_height = 100;

% Compute the position of the main rectangle to center it
rect_x = max(1, floor((width - rect_width) / 2));
rect_y = max(1, floor((height - rect_height) / 2));

% Create a black image
image2 = zeros(height, width);

% Ensure rectangle fits within the image boundaries
rect_x_end = min(width, rect_x + rect_width - 1);
rect_y_end = min(height, rect_y + rect_height - 1);

% Draw the main rectangle
image2(rect_y:rect_y_end, rect_x:rect_x_end) = 180;

% Define the size of the corner rectangles
corner_rect_width = 50;
corner_rect_height = 25;

% Define margins from the borders
xmargin = 480;
ymargin = 280;


% Top-left corner
corner_x1 = xmargin;
corner_y1 = ymargin;
corner_x1_end = min(width, corner_x1 + corner_rect_width - 1);
corner_y1_end = min(height, corner_y1 + corner_rect_height - 1);
image2(corner_y1:corner_y1_end, corner_x1:corner_x1_end) = 255;

% Top-right corner
corner_x2 = width - corner_rect_width - xmargin + 1;
corner_y2 = ymargin;
corner_x2_end = min(width, corner_x2 + corner_rect_width - 1);
corner_y2_end = min(height, corner_y2 + corner_rect_height - 1);
image2(corner_y2:corner_y2_end, corner_x2:corner_x2_end) = 255;

% Bottom-left corner
corner_x3 = xmargin;
corner_y3 = height - corner_rect_height - ymargin + 1;
corner_x3_end = min(width, corner_x3 + corner_rect_width - 1);
corner_y3_end = min(height, corner_y3 + corner_rect_height - 1);
image2(corner_y3:corner_y3_end, corner_x3:corner_x3_end) = 255;

% Bottom-right corner
corner_x4 = width - corner_rect_width - xmargin + 1;
corner_y4 = height - corner_rect_height - ymargin + 1;
corner_x4_end = min(width, corner_x4 + corner_rect_width - 1);
corner_y4_end = min(height, corner_y4 + corner_rect_height - 1);
image2(corner_y4:corner_y4_end, corner_x4:corner_x4_end) = 255;


% Display the image
imwrite(image2, 'projected_ref.jpg')

