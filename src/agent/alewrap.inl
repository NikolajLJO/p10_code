/* Copyright 2014 Google Inc.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

ALEInterface *getInterface();

void loadROM(const char *rom_file);

void setInt(const char *key, const int value);

void setBool(const char *key, const bool value);

void setFloat(const char *key, const float value);

bool game_over();

void reset_game();

reward_t act(Action action);

void fillRgbFromPalette(uint8_t *rgb, size_t rgb_size);

void fillGrayscaleFromPalette(uint8_t *gs, size_t gs_size);

int numLegalActions();

void legalActions(int *actions, size_t size);

int get_lives();
