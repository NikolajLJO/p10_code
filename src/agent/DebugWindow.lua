local debugWindow = torch.class('dqn.DebugWindow')


function debugWindow:__init(args)

    self.width                      = args.width
    self.height                     = args.height
    self.caption                    = args.caption or ""
    self.img                        = torch.DoubleTensor(3, args.height, args.width):zero()
    self.window                     = nil
    self.text_x                     = 10
    self.text_y_start               = 10
    self.text_y_increment           = 10
    self.text_y                     = self.text_y_start
    self.text_size                  = 1
    self.color                      = {0, 255, 0}
end


function debugWindow:clear()
    self.img:zero()
    self.text_y = self.text_y_start
end


function debugWindow:add_text(text)
    self.img = image.drawText(self.img, text, self.text_x, self.text_y, {color=self.color, size=self.text_size})
    self.text_y = self.text_y + self.text_y_increment
end


function debugWindow:repaint()
    self.window = image.display({win=self.window, image=self.img, legend=self.caption})
end