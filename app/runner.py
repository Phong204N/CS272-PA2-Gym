import myenv

import random
from PIL import Image, ImageDraw
from datetime import datetime
from pathlib import Path

CONST_SEED = None
CONST_TILE_SIZE = 10
def timestamp() -> str:
    return datetime.now().strftime(r'%d%b%Y-%H%M%S')
def rect_tl_to_br(tl:tuple, size:tuple) -> tuple:
    return (tl[0]+size[0], tl[1]+size[1])
def random_rgb() -> tuple:
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))
def render_to_image(render:str) -> Image.Image:
    if output[step].strip().upper() == "R":
        return flash_screen((128,0,0))
    elif output[step].strip().upper() == "G":
        return flash_screen((128,0,0))
    result = Image.new("RGB", (CONST_TILE_SIZE * (env.CONST_WORLD_X+2), CONST_TILE_SIZE * (env.CONST_WORLD_Y+2)))
    canvas = ImageDraw.Draw(result)
    cursor = (0,0)
    for char in render:
        color = (0,0,0)
        char = char.upper()
        if (char in ['╔','═','╗','║','╚','╝']):
            color = (255,255,255)
        elif (char in ['.']):
            color = (0,0,0)
        elif (char in ['H']):
            color = (0,255,128)
        elif (char in ['B','T']):
            color = (0,255,0)
        elif (char in ['@']):
            color = (255,0,0)
        elif (char in ['\n', '\r']):
            cursor = (0, cursor[1] + CONST_TILE_SIZE)
            continue
        canvas.rectangle([cursor, rect_tl_to_br(cursor, (CONST_TILE_SIZE,CONST_TILE_SIZE))], fill=color)
        cursor = (cursor[0]+ CONST_TILE_SIZE, cursor[1])
    return result
def flash_screen(color:tuple=(0,0,0)) -> Image.Image:
    result = Image.new("RGB", (CONST_TILE_SIZE * (env.CONST_WORLD_X+2), CONST_TILE_SIZE * (env.CONST_WORLD_Y+2)))
    canvas = ImageDraw.Draw(im)
    canvas.rectangle([(0,0),(im.width,im.height)], fill=color)
    return result
def im_array_to_gif(im_array:list, title:str, frame_duration:int=200) -> Path:
    im_array[0].save(
        title,
        save_all=True,
        append_images=im_array[1:],
        duration=frame_duration,
        loop=0
    )

env = myenv.MyEnv(render_mode="ansi")
running = True
env.reset(seed=CONST_SEED)
output = []
output.append(env.render())
while running:
    action = env.np_random.integers(0,env.action_space.n)
    obs, reward, terminated, truncated, info = env.step(action)
    output.append(env.render())
    if terminated or truncated:
        if len(info.get("snake_deque")) < 5:
            env.reset(seed=CONST_SEED)
            output.append("R")
            # output = []
        else:
            output.append("G")
            output.append("G")
            output.append("G")
            output.append("G")
            output.append("G")
            running = False

im_array:list[Image.Image] = []
for step in range(len(output)):
    im = render_to_image(output[step])
    im_array.append(im)

# im_array_to_gif(
#     im_array=im_array, 
#     title=f"snake_rand_{env.np_random_seed}.gif"
# )

im_array_to_gif(im_array, f"snake_rand_comp_{timestamp()}.gif", 20)