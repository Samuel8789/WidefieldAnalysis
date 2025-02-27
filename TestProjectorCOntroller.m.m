pkg load psychtoolbox; % Load Psychtoolbox
pkg load image; % For image display

function update_gui_display(ax, image_data, intensity)
  % Update GUI plot with the current image
  image(ax, image_data * intensity / 255); % Scale intensity
  colormap(ax, gray); % Grayscale colormap
  axis(ax, 'off'); % Hide axis
  drawnow;
endfunction

function update_intensity(hObject, event, win, texture1, texture2, ax)
  intensity = get(hObject, 'Value'); % Get intensity from slider
  ProjectedColor = [0 intensity 0]; % Apply new intensity (green)

  % Store intensity in button
  btn = findobj('Tag', 'ToggleButton');
  set(btn, 'UserData', struct('state', get(btn, 'UserData').state, 'intensity', intensity));

  % Refresh current screen with new intensity
  screen_state = get(btn, 'UserData').state;
  image_data = ones(720, 1280) * (screen_state * 255); % White if state=1, Black if state=0
  update_gui_display(ax, image_data, intensity);

  if screen_state
    Screen('DrawTexture', win, texture1, [], [], [], [], [], ProjectedColor);
  else
    Screen('DrawTexture', win, texture2, [], [], [], [], [], ProjectedColor);
  end
  Screen('Flip', win);
endfunction

function toggle_screen(hObject, event, win, texture1, texture2, ax)
  data = get(hObject, 'UserData'); % Get current state and intensity
  data.state = ~data.state; % Toggle screen state
  set(hObject, 'UserData', data); % Store new state

  ProjectedColor = [0 data.intensity 0]; % Use stored intensity
  image_data = ones(720, 1280) * (data.state * 255); % White if state=1, Black if state=0
  update_gui_display(ax, image_data, data.intensity);

  if data.state
    Screen('DrawTexture', win, texture1, [], [], [], [], [], ProjectedColor); % White screen
  else
    Screen('DrawTexture', win, texture2, [], [], [], [], [], ProjectedColor); % Black screen
  end
  Screen('Flip', win);
endfunction

function create_gui()
  % GUI stays on primary screen
  gui_screenid = min(Screen('Screens'));
  fig = figure('Position', [50, 50, 500, 300], 'MenuBar', 'none', 'Name', 'Screen Control', 'NumberTitle', 'off');

  % Psychtoolbox Setup (Fullscreen on secondary screen)
  PsychDefaultSetup(1);
  Screen('Preference', 'SkipSyncTests', 1);
  AssertOpenGL;

  exp_screenid = max(Screen('Screens'));
  [win, rect] = Screen('OpenWindow', exp_screenid, [0 0 0]); % Fullscreen mode

  % Image Setup
  width = 1280;
  height = 720;
  image1 = ones(height, width) * 255; % White
  image2 = ones(height, width) * 0;   % Black
  texture1 = Screen('MakeTexture', win, image1);
  texture2 = Screen('MakeTexture', win, image2);

  % Initial intensity
  intensity = 45;
  ProjectedColor = [0 intensity 0];

  % Start with white screen
  Screen('DrawTexture', win, texture1, [], [], [], [], [], ProjectedColor);
  Screen('Flip', win);

  % Axes for live image plot
  ax = axes('Parent', fig, 'Position', [0.1, 0.4, 0.8, 0.5]);
  update_gui_display(ax, image1, intensity);

  % Toggle button
  btn = uicontrol(fig, 'Style', 'pushbutton', 'String', 'Toggle Screen', ...
                  'Position', [50, 100, 200, 50], ...
                  'Callback', {@toggle_screen, win, texture1, texture2, ax}, ...
                  'Tag', 'ToggleButton');
  set(btn, 'UserData', struct('state', 1, 'intensity', intensity));

  % Intensity slider
  slider = uicontrol(fig, 'Style', 'slider', 'Min', 0, 'Max', 255, 'Value', intensity, ...
                     'Position', [50, 50, 200, 30], ...
                     'Callback', {@update_intensity, win, texture1, texture2, ax});
endfunction

create_gui();

