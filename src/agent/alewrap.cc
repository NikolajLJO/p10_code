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


#include "alewrap.h"

/*
#include <common/Array.hxx>

#include <stdexcept>
#include <cassert>
#include <algorithm>
*/

ALEInterface aleInterface;
std::vector<unsigned char> screen_RGB;
std::vector<unsigned char> screen_gs;

ALEInterface *getInterface() {
    return &aleInterface;
}

void loadROM(const char *rom_file) {
    aleInterface.loadROM(rom_file);
}

void setInt(const char *key, const int value) {
    aleInterface.setInt(key, value);
}

void setBool(const char *key, const bool value)
{
    aleInterface.setBool(key, value);
}

void setFloat(const char *key, const float value)
{
    aleInterface.setFloat(key, value);
}

bool game_over()
{
    return aleInterface.game_over();
}

void reset_game()
{
    aleInterface.reset_game();
}

reward_t act(Action action)
{
    aleInterface.act(action);
}

void fillRgbFromPalette(uint8_t *rgb, size_t rgb_size)
{
    assert(rgb_size >= 0);

    aleInterface.getScreenRGB(screen_RGB);

    for (int i = 0; (3 * i) < rgb_size; i++)
    {
        rgb[i] = screen_RGB[3 * i];
        rgb[i + (rgb_size / 3)] = screen_RGB[3 * i + 1];
        rgb[i + 2 * (rgb_size / 3)] = screen_RGB[3 * i + 2];
    }
}

void fillGrayscaleFromPalette(uint8_t *gs, size_t gs_size)
{
    assert(gs_size >= 0);

    aleInterface.getScreenGrayscale(screen_gs);

    for (int i = 0; i < gs_size; i++)
    {
        gs[i] = screen_gs[i];
    }
}

// Returns the number of legal actions
int numLegalActions()
{
    ActionVect actionVect = aleInterface.getLegalActionSet();
    return actionVect.size();
}

// Returns the valid actions for a game
void legalActions(int *actions, size_t size)
{
    ActionVect actionVect = aleInterface.getLegalActionSet();
    int numLegalActions = actionVect.size();

    for (int i = 0; i < numLegalActions; i++)
    {
        actions[i] = actionVect[i];
    }
}

int get_lives()
{
    return aleInterface.lives();
}
