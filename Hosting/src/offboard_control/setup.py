from setuptools import find_packages, setup

package_name = 'offboard_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dz',
    maintainer_email='dz@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'offboard_control = offboard_control.offboard_control:main',
            'offboard_control_1 = offboard_control.offboard_control_1:main',
            'offboard_control_2 = offboard_control.offboard_control_2:main',
            'offboard_control_fuzzy = offboard_control.offboard_control_fuzzy:main',
            'offboard_control_fxtsmc = offboard_control.offboard_control_fxtsmc:main',
        ],
    },
)
