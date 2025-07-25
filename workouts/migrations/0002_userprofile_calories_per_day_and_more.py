# Generated by Django 5.2.4 on 2025-07-13 17:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('workouts', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='calories_per_day',
            field=models.IntegerField(blank=True, help_text='Daily calorie intake', null=True),
        ),
        migrations.AddField(
            model_name='userprofile',
            name='equipment_available',
            field=models.TextField(blank=True, help_text='Available equipment (comma-separated)'),
        ),
        migrations.AddField(
            model_name='userprofile',
            name='weak_muscles',
            field=models.CharField(blank=True, help_text='Comma-separated list of weak muscle groups', max_length=200),
        ),
        migrations.AlterField(
            model_name='userprofile',
            name='primary_goal',
            field=models.CharField(choices=[('weight_loss', 'Weight Loss'), ('muscle_gain', 'Muscle Gain'), ('strength', 'Build Strength'), ('endurance', 'Improve Endurance'), ('toning', 'Toning & Definition'), ('general_fitness', 'General Fitness'), ('bulking', 'Bulking'), ('cutting', 'Cutting'), ('maintaining', 'Maintaining')], max_length=20),
        ),
    ]
