'''
v2.0
First attempt at using a GAN agent. This version will most likely just be
functional testing.
'''

import math, sys
import numpy as np
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS
from lux import annotate
import logging
import random


logging.basicConfig(filename='agent.log', level=logging.INFO)

DIRECTIONS = Constants.DIRECTIONS
game_state = None


def get_tiles(game_state, width, height):
    resource_tiles: list[Cell] = []
    empty_tiles: list[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)
            elif not cell.citytile:
                empty_tiles.append(cell)
    return resource_tiles, empty_tiles


def get_closest_resource(unit, resource_tiles, player):
    closest_dist = math.inf
    closest_resource_tile = None
    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
    for resource_tile in resource_tiles:
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile
    return closest_resource_tile


def get_closest_empty(unit, empty_tiles, player):
    closest_dist = math.inf
    closest_empty_tile = None
    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
    for empty_tile in empty_tiles:
        dist = empty_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_empty_tile = empty_tile
    return closest_empty_tile


def get_closest_city(player, unit):
    closest_dist = math.inf
    closest_city_tile = None
    for k, city in player.cities.items():
        for city_tile in city.citytiles:
            dist = city_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_city_tile = city_tile
                closest_city = city
    return closest_city, closest_city_tile


def get_closest_unprepared_city(player, unit):
    closest_dist = math.inf
    closest_city_tile = None
    for k, city in player.cities.items():
        if city.fuel < city.get_light_upkeep()*10:
            for city_tile in city.citytiles:
                dist = city_tile.pos.distance_to(unit.pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_city_tile = city_tile
                    closest_city = city
    return closest_city, closest_city_tile


def get_unprepared_cities(player):
    closest_dist = math.inf
    closest_city_tile = None
    unprepared_cities = []
    for k, city in player.cities.items():
        if city.fuel < city.get_light_upkeep()*10:
            unprepared_cities.append(city)
    return unprepared_cities


def get_move_step(unit_pos, destination):
    dir_diff = (destination.x - unit_pos.x, destination.y - unit_pos.y)
    xdiff = dir_diff[0]
    ydiff = dir_diff[1]
    if abs(ydiff) > abs(xdiff):
        target_cell = game_state.map.get_cell(unit_pos.x, unit_pos.y+np.sign(ydiff))
        if not target_cell.citytile:
            if np.sign(ydiff) == 1:
                return 's'
            else:
                return 'n'
        else:
            if np.sign(xdiff) == 1:
                return 'e'
            else:
                return 'w'
    else:
        target_cell = game_state.map.get_cell(unit_pos.x+np.sign(xdiff), unit_pos.y)
        if not target_cell.citytile:
            if np.sign(xdiff) == 1:
                return 'e'
            else:
                return 'w'
        else:
            if np.sign(ydiff) == 1:
                return 's'
            else:
                return 'n'


def collision_detection(unit_id, unit_positions):
    active_pos = unit_positions[unit_id]
    for k, v in unit_positions.items():
        if v == active_pos and k != unit_id:
            return True
    return False


def get_unit_positions(player):
    unit_positions = {}
    for unit in player.units:
        unit_positions[unit.id] = unit.pos
    return unit_positions


def agent(observation, configuration):
    global game_state

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    resource_tiles, empty_tiles = get_tiles(game_state, width, height)
    
    # we iterate over all our units and do something with them
    for unit in player.units:
        unit_positions = get_unit_positions(player)
        if unit.is_worker() and unit.can_act():
            if collision_detection(unit.id, unit_positions):
                actions.append(unit.move(random.choice(('n','s','e','w'))))
                continue
            if unit.get_cargo_space_left() > 0:
                closest_resource_tile = get_closest_resource(unit, resource_tiles, player)
                if closest_resource_tile is not None:
                    logging.info(f'Turn {game_state.turn}. Worker is moving to resource. Extra space {unit.get_cargo_space_left()}...')
                    move_dir = get_move_step(unit.pos, closest_resource_tile.pos)
                    actions.append(unit.move(move_dir))
            else:
                # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                if len(player.cities) > 0 and len(get_unprepared_cities(player)) > 0:
                    closest_city, closest_city_tile = get_closest_unprepared_city(player, unit)
                    if closest_city_tile is not None:
                        logging.info(f'Turn {game_state.turn}. Worker is moving to city...')
                        move_dir = unit.pos.direction_to(closest_city_tile.pos)
                        actions.append(unit.move(move_dir))
                    else:
                        if not game_state.map.get_cell(unit.pos.x, unit.pos.y).has_resource():
                            logging.info(f'Turn {game_state.turn}. Worker is building a city...')
                            actions.append(unit.build_city())
                        else:
                            closest_empty_tile = get_closest_empty(unit, empty_tiles, player)
                            if closest_empty_tile is not None:
                                logging.info(f'Turn {game_state.turn}. Worker is moving to empty tile...')
                                move_dir = get_move_step(unit.pos, closest_empty_tile.pos)
                                actions.append(unit.move(move_dir))
                else:
                    if not game_state.map.get_cell(unit.pos.x, unit.pos.y).has_resource():
                        logging.info(f'Turn {game_state.turn}. Worker is building a city...')
                        actions.append(unit.build_city())
                    else:
                        closest_empty_tile = get_closest_empty(unit, empty_tiles, player)
                        if closest_empty_tile is not None:
                            logging.info(f'Turn {game_state.turn}. Worker is moving to empty tile...')
                            move_dir = get_move_step(unit.pos, closest_empty_tile.pos)
                            actions.append(unit.move(move_dir))
    for k, city in player.cities.items():
        for tile in city.citytiles:
            if tile.can_act():
                if len(player.units) < len(player.cities):
                    actions.append(tile.build_worker())
                else:
                    actions.append(tile.research())

    return actions
