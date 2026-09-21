import myenv

env = myenv.MyEnv(render_mode="ansi")
running = True
env.reset()
output = env.render()
while running:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    output += "\n" + env.render()
    if terminated or truncated:
        if len(info.get("snake_deque")) < 5:
            env.reset()
            output = env.render()
        else:
            running = False
            print(output)
        # running = False
        # print(output)