clear

addpath('./util')

det_list = '';
img_dir = '';
anns_dir = '';
bbox_dir = '';

bbox_topK = 30;

fid = fopen(det_list);

tline = fgetl(fid);
while ischar(tline)
  disp(tline)

  % Image
  figure(1)
  img = imread([img_dir '/' tline '.JPEG']);
  imshow(img)
  th = title(sprintf('%s', tline));
  set(th, 'Interpreter', 'none')
  hold on

  % Proposals
  if ~ isempty(bbox_dir)
    load([bbox_dir '/' tline '.mat']);
    boxes = boxes(1:bbox_topK, :);
    for i = 1:size(boxes, 1)
      x = boxes(i, 1);
      y = boxes(i, 2);
      w = boxes(i, 3) - boxes(i, 1) + 1;
      h = boxes(i, 4) - boxes(i, 2) + 1;
      rectangle('Position', [x y w h], 'EdgeColor', 'b');
    end
  end

  % Annotations
  anns = xml2struct(fullfile(anns_dir, [tline '.xml']));
  if isfield(anns.annotation, 'object')
    for k = 1:numel(anns.annotation.object)
      if numel(anns.annotation.object) == 1
        obj = anns.annotation.object;
      else
        obj = anns.annotation.object{k};
      end
      a_x = str2double(obj.bndbox.xmin.Text);
      a_y = str2double(obj.bndbox.ymin.Text);
      a_w = str2double(obj.bndbox.xmax.Text) - str2double(obj.bndbox.xmin.Text);
      a_h = str2double(obj.bndbox.ymax.Text) - str2double(obj.bndbox.ymin.Text);
      rectangle('Position', [a_x a_y a_w a_h], 'EdgeColor', 'r', 'LineWidth', 3);
      text(a_x, a_y - 6, obj.name.Text, 'Color', 'white', 'FontSize', 14);
    end
  end
  hold off

  waitforbuttonpress
  tline = fgetl(fid);
end

fclose(fid);
